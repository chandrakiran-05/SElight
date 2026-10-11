"""Paired COCO AP bootstrap by resampling images (or declared scene clusters).

Reuses class/IoU matches, then recomputes dataset-level score ranking, precision
envelopes and AP in every resample. Does not average per-image AP values.
"""
import argparse
import contextlib
import copy
import io
from pathlib import Path
import numpy as np
from selight.core import read_json,write_json,sha,NAMES

def evaluator(records,predictions):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    data=dict(info={},licenses=[],images=[],annotations=[],categories=[dict(id=i,name=n) for i,n in enumerate(NAMES)])
    detections=[]
    for i,r in enumerate(records):
        data['images'].append(dict(id=i,width=r['width'],height=r['height']))
        for c,x1,y1,x2,y2 in r['boxes']:
            data['annotations'].append(dict(id=len(data['annotations'])+1,image_id=i,category_id=int(c),bbox=[x1,y1,x2-x1,y2-y1],area=(x2-x1)*(y2-y1),iscrowd=0))
        for c,s,x1,y1,x2,y2 in predictions[r['id']]:
            detections.append(dict(image_id=i,category_id=int(c),score=s,bbox=[x1,y1,x2-x1,y2-y1]))
    with contextlib.redirect_stdout(io.StringIO()):
        gt=COCO(); gt.dataset=data; gt.createIndex()
        if detections: dt=gt.loadRes(detections)
        else: dt=COCO(); dt.dataset={**data,'annotations':[]}; dt.createIndex()
        ev=COCOeval(gt,dt,'bbox'); ev.params.imgIds=list(range(len(records))); ev.evaluate()
    return ev

def resampled_ap(ev,indices):
    new=copy.copy(ev); new.params=copy.deepcopy(ev.params)
    new.params.imgIds=list(range(len(indices))); new._paramsEval=copy.deepcopy(new.params)
    n=len(ev.params.imgIds); k=len(ev.params.catIds); a=len(ev.params.areaRng)
    new.evalImgs=[ev.evalImgs[(cat*a+area)*n+int(i)] for cat in range(k) for area in range(a) for i in indices]
    with contextlib.redirect_stdout(io.StringIO()): new.accumulate()
    precision=new.eval['precision'][:,:,:,0,-1]; valid=precision[precision>=0]
    return float(valid.mean()) if len(valid) else None

def main():
    p=argparse.ArgumentParser(); p.add_argument('--manifest',required=True); p.add_argument('--baseline',required=True)
    p.add_argument('--candidate',required=True); p.add_argument('--out',required=True); p.add_argument('--resamples',type=int,default=2000)
    p.add_argument('--cluster-field'); a=p.parse_args()
    if a.resamples<1: p.error('resamples must be positive')
    obj=read_json(a.manifest); b=read_json(a.baseline); c=read_json(a.candidate)
    if set(b)!=set(c): raise ValueError('Prediction IDs differ')
    rows=[r for r in obj['records'] if r['id'] in b]
    if len(rows)!=len(b): raise ValueError('Unknown IDs')
    units={}
    for i,r in enumerate(rows):
        key=r[a.cluster_field] if a.cluster_field else r['id']
        units.setdefault(str(key),[]).append(i)
    groups=list(units.values()); rng=np.random.default_rng(42)
    be=evaluator(rows,b); ce=evaluator(rows,c); base=resampled_ap(be,range(len(rows))); cand=resampled_ap(ce,range(len(rows)))
    if base is None or cand is None: raise ValueError('No eligible annotations')
    deltas=[]; undefined=0
    for _ in range(a.resamples):
        indices=[i for g in rng.integers(0,len(groups),len(groups)) for i in groups[g]]
        bp=resampled_ap(be,indices); cp=resampled_ap(ce,indices)
        if bp is None or cp is None: undefined+=1
        else: deltas.append(cp-bp)
    if not deltas: raise ValueError('All resamples lack annotations')
    write_json(a.out,dict(baseline_map=base,candidate_map=cand,difference=cand-base,
        percentile_95_interval=np.percentile(deltas,[2.5,97.5]).tolist(),resamples=a.resamples,valid_resamples=len(deltas),undefined_resamples=undefined,
        independent_units=len(groups),images=len(rows),cluster_field=a.cluster_field,seed=42,manifest_hash=sha(a.manifest),baseline_hash=sha(a.baseline),candidate_hash=sha(a.candidate),
        scope='Paired COCO AP50:95 recomputation, maxDet=100, original annotation areas. Percentile interval; conditional on supplied independent units; timing repeats are not accuracy samples.'))

if __name__=='__main__': main()
