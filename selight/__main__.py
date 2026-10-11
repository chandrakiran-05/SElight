import argparse
import json
from .core import read_json

def main():
    p=argparse.ArgumentParser(description='Selective low-light enhancement prototype. See README.md.')
    sub=p.add_subparsers(dest='command',required=True)
    s=sub.add_parser('prepare'); s.add_argument('--images',required=True); s.add_argument('--annotations',required=True)
    s.add_argument('--split-file',required=True); s.add_argument('--out',required=True)
    s.add_argument('--origin',type=int,choices=[0,1],required=True,help='Verify annotation source and preview boxes; do not guess.')
    s.add_argument('--deduplicate',action='store_true',help='Explicitly permit documented train/val/test-priority exact deduplication.')
    s=sub.add_parser('audit'); s.add_argument('--manifest',required=True)
    s=sub.add_parser('preview'); s.add_argument('--manifest',required=True); s.add_argument('--out',required=True); s.add_argument('--count',type=int,default=12)
    s=sub.add_parser('cache'); s.add_argument('--manifest',required=True); s.add_argument('--config',default='configs/default.json'); s.add_argument('--out',required=True); s.add_argument('--limit',type=int)
    s=sub.add_parser('train'); s.add_argument('--cache',required=True); s.add_argument('--out',required=True)
    s.add_argument('--ablation',choices=['full','no_cost','no_margin','brightness_only','no_clahe'],default='full')
    s=sub.add_parser('evaluate'); s.add_argument('--manifest',required=True); s.add_argument('--config',default='configs/default.json')
    s.add_argument('--policy',required=True); s.add_argument('--out',required=True); s.add_argument('--split',choices=['val','test'],default='test')
    s.add_argument('--repeats',type=int,default=3); s.add_argument('--limit',type=int)
    s=sub.add_parser('demo'); s.add_argument('--config',default='configs/default.json'); s.add_argument('--policy',default='models/policy.json')
    s.add_argument('--source',required=True); s.add_argument('--out',required=True)
    sub.add_parser('doctor')
    a=p.parse_args()
    if getattr(a,'limit',None) is not None and a.limit<1: p.error('limit must be positive')
    if getattr(a,'repeats',1)<1: p.error('repeats must be positive')
    if a.command=='prepare':
        from .data import prepare
        prepare(a.images,a.annotations,a.split_file,a.out,a.origin,a.deduplicate)
    elif a.command=='audit':
        from .data import audit
        print(json.dumps(audit(read_json(a.manifest)),indent=2))
    elif a.command=='preview':
        from .data import preview
        preview(a.manifest,a.out,a.count)
    elif a.command=='cache':
        from .experiment import build_cache
        build_cache(a.manifest,read_json(a.config),a.out,a.limit)
    elif a.command=='train':
        from .experiment import train
        train(a.cache,a.out,a.ablation)
    elif a.command=='evaluate':
        from .experiment import evaluate
        evaluate(a.manifest,read_json(a.config),a.policy,a.out,a.split,a.repeats,a.limit)
    elif a.command=='demo':
        from .experiment import demo
        demo(read_json(a.config),a.policy,a.source,a.out)
    elif a.command=='doctor':
        from .experiment import environment
        print(json.dumps(environment({}),indent=2))
        try:
            import torch
            print('CUDA available:',torch.cuda.is_available())
            if torch.cuda.is_available(): print('GPU:',torch.cuda.get_device_name(0))
        except ImportError: print('PyTorch is not installed. Core tests still run; detector commands need it.')

if __name__=='__main__': main()
