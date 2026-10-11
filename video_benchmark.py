"""Local-file video processing timing, excluding drawing/writing and camera/display.

Frames are processed independently; repeated frames never count as independent
accuracy samples. The supplied project clip is an ExDark image slideshow.
"""
import argparse
import csv
import time
from pathlib import Path
import cv2
import numpy as np
from selight.core import read_json,write_json,sha,Enhancer,select
from selight.detector import Detector
from selight.experiment import signature,environment,warmup

def main():
    p=argparse.ArgumentParser(); p.add_argument('--source',required=True); p.add_argument('--policy',required=True)
    p.add_argument('--config',default='configs/gpu.json'); p.add_argument('--out',required=True); a=p.parse_args()
    cfg=read_json(a.config); policy=read_json(a.policy); det=Detector(cfg); enh=Enhancer(cfg)
    if signature(cfg,det.weight_hash)!=policy['signature']: raise ValueError('Policy mismatch')
    cap=cv2.VideoCapture(a.source); ok,im=cap.read()
    if not ok: raise ValueError('Video cannot be decoded')
    fps=cap.get(cv2.CAP_PROP_FPS); size=[im.shape[1],im.shape[0]]; cap.release(); warmup(det,enh,im)
    out=Path(a.out); out.mkdir(parents=True,exist_ok=False); logs=[]
    for repeat in range(3):
        cap=cv2.VideoCapture(a.source); index=0
        try:
            while True:
                det.sync(); start=time.perf_counter(); ok,image=cap.read(); decoded=time.perf_counter()
                if not ok: break
                route=select(image,'selector',cfg,policy); processed=enh.apply(image,route); pred=det.predict(processed); det.sync(); end=time.perf_counter()
                logs.append(dict(repeat=repeat,frame=index,route=route,decode_ms=(decoded-start)*1000,processing_ms=(end-decoded)*1000,total_ms=(end-start)*1000)); index+=1
        finally: cap.release()
    with (out/'timings.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(logs[0])); writer.writeheader(); writer.writerows(logs)
    values=[r['total_ms'] for r in logs]
    write_json(out/'summary.json',dict(source_hash=sha(a.source),source=str(a.source),source_size=size,source_playback_fps=fps,frames_per_repeat=[sum(r['repeat']==i for r in logs) for i in range(3)],
        mean_ms=float(np.mean(values)),p95_ms=float(np.percentile(values,95)),aggregate_fps=1000/float(np.mean(values)),policy_hash=sha(a.policy),environment=environment(cfg),weights_hash=det.weight_hash,
        scope='Serial local-file decode + feature/route/enhancement + synchronized detector. Excludes acquisition, display, drawing, encoding, output writing and startup. Not capture-to-display end-to-end latency.'))

if __name__=='__main__': main()
