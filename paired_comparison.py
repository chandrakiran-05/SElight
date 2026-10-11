"""Paired image bootstrap for F1 utility, not an AP confidence interval."""
import argparse
import numpy as np
from selight.core import read_json, write_json
from selight.metrics import quality

def main():
    p=argparse.ArgumentParser(); p.add_argument('--manifest',required=True)
    p.add_argument('--baseline',required=True); p.add_argument('--candidate',required=True)
    p.add_argument('--out',required=True); p.add_argument('--confidence',type=float,default=.25)
    a=p.parse_args(); obj=read_json(a.manifest); b=read_json(a.baseline); c=read_json(a.candidate)
    if set(b)!=set(c): raise ValueError('Prediction IDs differ')
    rows=[r for r in obj['records'] if r['id'] in b]
    if len(rows)!=len(b): raise ValueError('Unknown prediction IDs')
    delta=np.array([quality(r['boxes'],c[r['id']],a.confidence)-quality(r['boxes'],b[r['id']],a.confidence) for r in rows])
    rng=np.random.default_rng(42)
    boot=[float(rng.choice(delta,size=len(delta),replace=True).mean()) for _ in range(2000)]
    write_json(a.out,{'mean_per_image_f1_difference':float(delta.mean()),'bootstrap_95_percent_interval':np.percentile(boot,[2.5,97.5]).tolist(),
          'samples':len(rows),'resamples':2000,'seed':42,'scope':'Independent-image bootstrap of macro per-image F1 difference, not COCO AP; correlated video frames require sequence bootstrap.'})

if __name__=='__main__': main()
