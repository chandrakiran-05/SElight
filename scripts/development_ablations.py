"""Train-only fitting, validation-only comparison using unchanged three-route cache."""
import contextlib
import argparse
import io
from pathlib import Path
import collections
import numpy as np
from selight.core import read_json,write_json,tree_route,ROUTES,sha
from selight.experiment import train,route_label
from selight.metrics import summary

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache',required=True); p.add_argument('--manifest',required=True); p.add_argument('--out',required=True)
    args=p.parse_args()
    cache=read_json(args.cache); obj=read_json(args.manifest)
    if cache['manifest_hash']!=sha(args.manifest): raise ValueError('Manifest mismatch')
    out=Path(args.out); out.mkdir(parents=True,exist_ok=False)
    byid={r['id']:r for r in obj['records']}; val=[r for r in cache['records'] if r['split']=='val']
    gt=[byid[r['id']] for r in val]; results={}
    for ablation in ['full','no_cost','no_margin','brightness_only','no_clahe']:
        dest=out/f'{ablation}.json'
        with contextlib.redirect_stdout(io.StringIO()): train(args.cache,dest,ablation)
        policy=read_json(dest)
        routes=[tree_route(policy['tree'],r['features'][:1] if ablation=='brightness_only' else r['features']) for r in val]
        predictions={r['id']:r['predictions'][a] for r,a in zip(val,routes)}
        with contextlib.redirect_stdout(io.StringIO()): result=summary(gt,predictions)
        base=np.array([r['quality']['original'] for r in val]); q=np.array([r['quality'][a] for r,a in zip(val,routes)])
        result.update(route_counts=dict(collections.Counter(routes)),improvement_rate=float(np.mean(q>base)),negative_transfer_rate=float(np.mean(q<base)),selection=policy['selection'],note='Cache-derived validation quality; no new deployment timing; test never used')
        results[ablation]=result; write_json(out/f'{ablation}_val_predictions.json',predictions)
    policy=read_json(out/'full.json')
    for method in ['original','gamma','clahe','threshold','brightness_tree','best_fixed']:
        routes=[]
        for r in val:
            routes.append(method if method in ROUTES else policy['best_fixed_route'] if method=='best_fixed' else tree_route(policy['brightness_tree'],r['features'][:1]) if method=='brightness_tree' else 'gamma' if r['features'][0]<policy['brightness_threshold'] else 'original')
        pred={r['id']:r['predictions'][a] for r,a in zip(val,routes)}
        with contextlib.redirect_stdout(io.StringIO()): result=summary(gt,pred)
        result['route_counts']=dict(collections.Counter(routes)); results[method]=result
        write_json(out/f'{method}_val_predictions.json',pred)
    write_json(out/'summary.json',dict(results=results,cache_hash=sha(args.cache),manifest_hash=sha(args.manifest),scope=f'{len(val)} validation images; same stored route predictions; no fresh inference or timing; fixed primary policy retained'))
    print({k:(v['map_50_95'],v['mean_image_f1'],v['route_counts']) for k,v in results.items()})

if __name__=='__main__': main()
