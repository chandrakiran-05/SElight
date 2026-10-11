"""Paired validation-only pretrained enhancer + identical frozen-detector pilot."""
import argparse
import csv
import time
from pathlib import Path
import cv2
import numpy as np
from selight.core import read_json,write_json,load_image,sha
from selight.data import audit
from selight.detector import Detector
from selight.external import ZeroDCEPP
from selight.experiment import records_for,environment,signature
from selight.metrics import summary

def main():
    p=argparse.ArgumentParser(); p.add_argument('--limit',type=int,default=30); p.add_argument('--out',required=True)
    p.add_argument('--manifest',required=True); p.add_argument('--config',default='configs/gpu.json')
    a=p.parse_args()
    if a.limit<1: p.error('limit must be positive')
    cfg=read_json(a.config); obj=read_json(a.manifest); audit(obj)
    rows=records_for(obj,'val',a.limit); det=Detector(cfg); ext=ZeroDCEPP(cfg['device'])
    out=Path(a.out); out.mkdir(parents=True,exist_ok=False)
    first=load_image(Path(obj['root'])/rows[0]['path'])
    for _ in range(10): det.predict(first); det.predict(ext.apply(first))
    det.sync(); rng=np.random.default_rng(42); logs=[]; results={}
    for rep in range(3):
        for method in rng.permutation(['original',ext.name]):
            predictions={}
            if cfg['device']!='cpu': det.torch.cuda.reset_peak_memory_stats()
            for r in rows:
                image=load_image(Path(obj['root'])/r['path'])
                det.sync(); begin=time.perf_counter()
                enhanced=ext.apply(image) if method==ext.name else image
                det.sync(); mid=time.perf_counter(); pred=det.predict(enhanced); det.sync(); end=time.perf_counter()
                predictions[r['id']]=pred
                logs.append(dict(method=str(method),repeat=rep,id=r['id'],enhance_ms=(mid-begin)*1000,detect_ms=(end-mid)*1000,processing_ms=(end-begin)*1000))
            write_json(out/f'{method}_repeat{rep}_predictions.json',predictions)
            result=summary(rows,predictions,cfg['operating_confidence'])
            times=[r['processing_ms'] for r in logs if r['method']==method and r['repeat']==rep]
            result.update(mean_ms=float(np.mean(times)),p95_ms=float(np.percentile(times,95)),fps=1000/float(np.mean(times)),cuda_peak_allocated_mb=det.torch.cuda.max_memory_allocated()/2**20 if cfg['device']!='cpu' else None)
            results[f'{method}_repeat{rep}']=result
    with (out/'timings.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(logs[0])); writer.writeheader(); writer.writerows(logs)
    write_json(out/'summary.json',dict(results=results,split='val',pilot_limit=a.limit,environment=environment(cfg),weights_hash=det.weight_hash,manifest_hash=sha(a.manifest),external_provenance=ext.provenance,
        adaptation='Official architecture and checkpoint; scale factor 12; RGB float [0,1]; replicate padding to multiples of 12; crop output, uint8 round conversion; preserves native input geometry instead of official input cropping',
        timing_scope='Processing only; enhancer transfers/quantization and detector; excludes decode/startup/serialization; compare only with processing-scope columns',
        checkpoint_config_signature=signature(cfg,det.weight_hash)))

if __name__=='__main__': main()
