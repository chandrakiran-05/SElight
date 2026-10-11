"""Strict ExDark conversion; preserve official split and audit image content."""
from pathlib import Path
import numpy as np
import cv2
from .core import load_image, sha, read_json, write_json

MAP = {"people":0,"bicycle":1,"car":2,"motorbike":3,"bus":4}
ALL_CLASSES = {"bicycle","boat","bottle","bus","car","cat","chair","cup","dog","motorbike","people","table"}

def parse_boxes(text, width, height, origin):
    boxes=[]
    for line in text.splitlines():
        parts=line.strip().split()
        if not parts or parts[0].startswith('%'): continue
        name=parts[0].lower()
        if name not in ALL_CLASSES: raise ValueError(f"Unknown annotation class/header: {line}")
        if len(parts)<5: raise ValueError(f"Short annotation: {line}")
        l,t,w,h=map(float,parts[1:5])
        if not np.isfinite([l,t,w,h]).all() or w<=0 or h<=0: raise ValueError(f"Invalid box: {line}")
        # Caller must explicitly choose source origin after inspecting the source release.
        x1=max(0.,l-origin); y1=max(0.,t-origin)
        x2=min(float(width),l-origin+w); y2=min(float(height),t-origin+h)
        if x2<=x1 or y2<=y1: raise ValueError(f"Box outside image: {line}")
        if name in MAP: boxes.append([MAP[name],x1,y1,x2,y2])
    return boxes

def index_files(root, suffixes):
    result={}
    for p in Path(root).rglob('*'):
        if p.is_file() and p.suffix.lower() in suffixes:
            k=p.name.lower()
            if k in result: raise ValueError(f"Ambiguous duplicate basename: {p.name}")
            result[k]=p
    return result

def prepare(images, annotations, split_file, out, origin, deduplicate=False):
    images=Path(images).resolve(); ann=index_files(annotations,{'.txt'})
    ims=index_files(images,{'.jpg','.jpeg','.png','.bmp'})
    raw_records=[]; seen=set()
    for line in Path(split_file).read_text(encoding='utf-8-sig').splitlines():
        p=line.split()
        if not p or p[0].startswith(('#','%')) or p[0].lower()=='name': continue
        if len(p)!=5: raise ValueError(f"Expected five metadata columns: {line}")
        name,cls,light,env,split=p
        try: split={1:'train',2:'val',3:'test'}[int(split)]
        except (ValueError,KeyError): raise ValueError(f"Invalid split: {line}")
        k=name.lower()
        if k in seen: raise ValueError(f"Duplicate metadata image {name}")
        seen.add(k)
        if k not in ims: raise FileNotFoundError(f"Image missing: {name}; do not silently drop it")
        raw_records.append((name, split, int(light), int(env), k))

    split_prio = {'train': 1, 'val': 2, 'test': 3}
    sha_map = {}
    omitted=[]
    for name, split, light, env, k in raw_records:
        img_sha = sha(ims[k])
        if img_sha not in sha_map:
            sha_map[img_sha] = (name, split, light, env, k)
        else:
            prev_name, prev_split, _, _, _ = sha_map[img_sha]
            if not deduplicate:
                raise ValueError('Exact duplicate source images; explicitly use --deduplicate to create a documented modified split')
            if split_prio[split] < split_prio[prev_split]:
                omitted.append({'omitted':prev_name,'split':prev_split,'retained':name,'sha256':img_sha})
                print(f"Resolving duplicate: replacing {prev_name} ({prev_split}) with {name} ({split})")
                sha_map[img_sha] = (name, split, light, env, k)
            else:
                omitted.append({'omitted':name,'split':split,'retained':prev_name,'sha256':img_sha})
                print(f"Resolving duplicate: omitting {name} ({split}) in favor of {prev_name} ({prev_split})")

    rows = []
    for img_sha, (name, split, light, env, k) in sha_map.items():
        candidates=[ann[a] for a in (k+'.txt',Path(k).stem+'.txt') if a in ann]
        candidates=list(dict.fromkeys(candidates))
        if len(candidates)!=1: raise ValueError(f"Missing/ambiguous annotation for {name}")
        im=load_image(ims[k]); h,w=im.shape[:2]
        boxes=parse_boxes(candidates[0].read_text(encoding='utf-8-sig'),w,h,origin)
        rows.append({'id':name,'path':ims[k].relative_to(images).as_posix(),'split':split,
                     'width':w,'height':h,'boxes':boxes,'sha256':img_sha,
                     'lighting':light,'environment':env})
    if not rows: raise ValueError('No data converted')
    obj={'schema':1,'root':str(images),'source':'ExDark official metadata; modified split after explicit deduplication' if omitted else 'ExDark official metadata splits',
         'source_split_hash':sha(split_file),'deduplication':{'rule':'prefer train, then val, then test; first metadata row within a split','omitted':omitted},
         'coordinate_origin':origin,'subset':'five COCO-overlap classes; negatives retained; omissions listed in deduplication',
         'records':rows}
    audit(obj,verify_files=False)
    write_json(out,obj)
    print(f'Wrote {len(rows)} images to {out}')

def audit(obj, verify_files=True):
    seen={}; ids=set(); report={s:{'images':0,'boxes':0,'class_counts':[0]*5} for s in ('train','val','test')}
    for r in obj['records']:
        if r['id'] in ids: raise ValueError('Duplicate image ID')
        ids.add(r['id']); s=r['split']
        if s not in report: raise ValueError('Unknown split')
        path=Path(obj['root'])/r['path']
        if verify_files and sha(path)!=r['sha256']: raise ValueError(f'Image changed: {path}')
        if r['sha256'] in seen and seen[r['sha256']] != s: raise ValueError('Exact duplicate crosses splits')
        seen[r['sha256']]=s
        for c,x1,y1,x2,y2 in r['boxes']:
            if int(c)!=c or not 0<=c<5 or not (0<=x1<x2<=r['width'] and 0<=y1<y2<=r['height']):
                raise ValueError(f'Invalid class/coordinates: {r["id"]}')
            report[s]['class_counts'][int(c)]+=1
        report[s]['images']+=1; report[s]['boxes']+=len(r['boxes'])
    for s,v in report.items():
        if not v['images']: raise ValueError(f'Empty {s} split')
    return report

def preview(manifest, out, count=12):
    obj=read_json(manifest); out=Path(out); out.mkdir(parents=True,exist_ok=True)
    # Spread inspection over the manifest rather than showing only its first class.
    for i in np.linspace(0,len(obj['records'])-1,min(count,len(obj['records'])),dtype=int):
        r=obj['records'][i]; im=load_image(Path(obj['root'])/r['path'])
        for c,x1,y1,x2,y2 in r['boxes']:
            cv2.rectangle(im,(round(x1),round(y1)),(round(x2),round(y2)),(0,255,0),2)
            cv2.putText(im,str(c),(round(x1),max(15,round(y1))),0,.5,(0,255,0),1)
        cv2.imwrite(str(out/f'{i:05d}.jpg'),im)
