import csv
import json
import platform
import time
from pathlib import Path
import importlib.metadata as metadata
import cv2
import numpy as np
from .core import ROUTES, FEATURES, Enhancer, features, select, tree_route, load_image, read_json, write_json, digest, sha
from .metrics import quality, summary
from .data import audit

def signature(cfg,weight_hash):
    return digest({k:v for k,v in cfg.items() if k not in ('device','threads','weights')}|{'weight_hash':weight_hash})

def environment(cfg):
    versions={}
    for name in ('numpy','opencv-python','opencv-python-headless','scikit-learn','torch','torchvision','ultralytics','pycocotools'):
        try: versions[name]=metadata.version(name)
        except metadata.PackageNotFoundError: pass
    info={'platform':platform.platform(),'python':platform.python_version(),'config':cfg,'versions':versions,
          'code_hashes':{p.name:sha(p) for p in Path(__file__).parent.glob('*.py')},'opencv_threads':cv2.getNumThreads()}
    try:
        import torch
        info.update(cuda_available=torch.cuda.is_available(),torch_cuda_build=torch.version.cuda,
                    torch_threads=torch.get_num_threads(),cudnn_benchmark=torch.backends.cudnn.benchmark,
                    cudnn_deterministic=torch.backends.cudnn.deterministic)
        if torch.cuda.is_available(): info['gpu']=torch.cuda.get_device_name(0)
    except ImportError: pass
    return info

def records_for(obj,split,limit=None):
    records=[r for r in obj['records'] if r['split']==split]
    # Stable hash sampling avoids first-N category bias. Limit is a pilot, not full benchmark.
    records=sorted(records,key=lambda r:digest(r['id']))
    return records[:limit] if limit else records

def warmup(det,enh,image,n=10):
    for r in ROUTES:
        for _ in range(n): det.predict(enh.apply(image,r))
    det.sync()

def build_cache(manifest,cfg,out,limit=None):
    from .detector import Detector
    obj=read_json(manifest); audit(obj)
    det=Detector(cfg); enh=Enhancer(cfg); rng=np.random.default_rng(cfg['seed'])
    cv2.setNumThreads(cfg['threads'])
    records=records_for(obj,'train',limit)+records_for(obj,'val',limit)
    warmup(det,enh,load_image(Path(obj['root'])/records[0]['path']))
    output=[]
    for i,r in enumerate(records):
        im=load_image(Path(obj['root'])/r['path']); scores={}; durations={}; predictions={}; timing_samples={}
        for route in rng.permutation(ROUTES):
            route=str(route); elapsed=[]
            for _ in range(3):
                det.sync(); start=time.perf_counter(); processed=enh.apply(im,route)
                pred=det.predict(processed); det.sync(); elapsed.append((time.perf_counter()-start)*1000)
            scores[route]=quality(r['boxes'],pred,cfg['operating_confidence'])
            durations[route]=float(np.median(elapsed)); timing_samples[route]=elapsed; predictions[route]=pred
        output.append({'id':r['id'],'sha256':r['sha256'],'split':r['split'],'features':features(im,cfg['feature_size']),
                       'quality':scores,'ms':durations,'timing_samples_ms':timing_samples,'predictions':predictions})
        if (i+1)%25==0: print(f'Cached {i+1}/{len(records)}')
    write_json(out,{'signature':signature(cfg,det.weight_hash),'weights_hash':det.weight_hash,'manifest_hash':sha(manifest),
      'environment':environment(cfg),'records':output,'scope':'development only; no test records',
      'pilot_limit_per_split':limit})

def route_label(q,cost,lam,margin=.02):
    # Retain original unless a route has a meaningful proxy-quality improvement.
    eligible=['original']+[r for r in ROUTES[1:] if q[r]>=q['original']+margin]
    return max(eligible,key=lambda r:(q[r]-lam*cost[r],-ROUTES.index(r)))

def serialize_tree(model):
    t=model.tree_
    return {'left':t.children_left.tolist(),'right':t.children_right.tolist(),
            'feature':t.feature.tolist(),'threshold':t.threshold.tolist(),
            'route':[str(model.classes_[int(np.argmax(v))]) for v in t.value[:,0,:]]}

def train(cache_path,out,ablation='full'):
    from sklearn.tree import DecisionTreeClassifier, export_text
    cache=read_json(cache_path)
    if any(r['split'] not in ('train','val') for r in cache['records']): raise ValueError('Test data in development cache')
    tr=[r for r in cache['records'] if r['split']=='train']; va=[r for r in cache['records'] if r['split']=='val']
    if not tr or not va: raise ValueError('Need nonempty train and val data')
    if {r['sha256'] for r in tr}&{r['sha256'] for r in va}: raise ValueError('Train/val leakage')
    if ablation not in ('full','no_cost','no_margin','brightness_only','no_clahe'):
        raise ValueError('Unknown ablation')
    for r in tr+va:
        if len(r['features'])!=8 or not np.isfinite(r['features']).all(): raise ValueError('Invalid features')
        if any(not np.isfinite(r['ms'][a]) or r['ms'][a]<=0 for a in ROUTES): raise ValueError('Invalid timing')
        if any(not np.isfinite(r['quality'][a]) or not 0<=r['quality'][a]<=1 for a in ROUTES): raise ValueError('Invalid quality')
    margin=0. if ablation=='no_margin' else .02
    val_weight=0. if ablation=='no_cost' else .01
    allowed=ROUTES[:2] if ablation=='no_clahe' else ROUTES
    base=float(np.median([r['ms']['original'] for r in tr]))
    # Pooled train costs reduce dependence on noisy individual timing samples.
    cost={r:max(0.,float(np.median([x['ms'][r] for x in tr]))/base-1) for r in ROUTES}; cost['original']=0.
    stats={'train':{},'val':{}}
    for key,rows in [('train',tr),('val',va)]:
        q=np.array([[r['quality'][a] for a in ROUTES] for r in rows])
        stats[key]={'mean_image_f1':dict(zip(ROUTES,q.mean(0).tolist())),
                    'oracle_mean_image_f1':float(q.max(1).mean()),
                    'oracle_gap_over_best_fixed':float(q.max(1).mean()-q.mean(0).max()),
                    'note':'Label-aware diagnostic upper bound only for mean per-image F1; not AP or deployable performance.'}
    threshold_rows=[]
    for threshold in (0.,.15,.25,.30,.40,.50,1.01):
        acts=['gamma' if r['features'][0]<threshold else 'original' for r in va]
        value=float(np.mean([r['quality'][a]-val_weight*cost[a] for r,a in zip(va,acts)]))
        threshold_rows.append({'threshold':threshold,'val_objective':value})
    best_threshold=max(threshold_rows,key=lambda r:r['val_objective'])['threshold']
    candidates=[]; best=None
    for depth in (2,3,4):
        for lam in ((0.,) if ablation=='no_cost' else (0.,.01,.05)):
            labels=[route_label({a:(r['quality'][a] if a in allowed else -1.) for a in ROUTES},cost,lam,margin) for r in tr]
            model=DecisionTreeClassifier(max_depth=depth,min_samples_leaf=max(2,int(.02*len(tr))),random_state=42)
            model.fit([r['features'][:1] if ablation=='brightness_only' else r['features'] for r in tr],labels)
            routes=model.predict([r['features'][:1] if ablation=='brightness_only' else r['features'] for r in va])
            mq=float(np.mean([r['quality'][a] for r,a in zip(va,routes)]))
            mc=float(np.mean([cost[a] for a in routes]))
            # Fixed validation cost weight across candidates; no comparing different objectives.
            value=mq-val_weight*mc
            row={'depth':depth,'training_lambda':lam,'val_mean_image_f1':mq,'val_extra_cost':mc,'val_objective':value}
            candidates.append(row)
            if best is None or value>best[0]+1e-12:
                best=(value,model,row)
    _,model,chosen=best
    checkpoint_hash=cache.get('weights_hash')
    checkpoint=cache['environment']['config'].get('weights')
    if checkpoint_hash is None and checkpoint and Path(checkpoint).is_file():
        observed_hash=sha(checkpoint)
        if signature(cache['environment']['config'],observed_hash)==cache['signature']: checkpoint_hash=observed_hash
    names=FEATURES[:1] if ablation=='brightness_only' else FEATURES
    brightness_candidates=[]; brightness_best=None
    for depth in (2,3,4):
        for lam in ((0.,) if ablation=='no_cost' else (0.,.01,.05)):
            labels=[route_label({a:(r['quality'][a] if a in allowed else -1.) for a in ROUTES},cost,lam,margin) for r in tr]
            bmodel=DecisionTreeClassifier(max_depth=depth,min_samples_leaf=max(2,int(.02*len(tr))),random_state=42)
            bmodel.fit([[r['features'][0]] for r in tr],labels)
            acts=bmodel.predict([[r['features'][0]] for r in va])
            mq=float(np.mean([r['quality'][a] for r,a in zip(va,acts)])); mc=float(np.mean([cost[a] for a in acts]))
            row={'depth':depth,'training_lambda':lam,'val_mean_image_f1':mq,'val_extra_cost':mc,'val_objective':mq-val_weight*mc}
            brightness_candidates.append(row)
            if brightness_best is None or row['val_objective']>brightness_best[0]+1e-12:
                brightness_best=(row['val_objective'],bmodel,row)
    fixed=[{'route':a,'val_objective':float(np.mean([r['quality'][a] for r in va]))-val_weight*cost[a]} for a in allowed]
    policy={'schema':1,'signature':cache['signature'],'training_manifest_hash':cache['manifest_hash'],
            'cache_hash':sha(cache_path),'feature_names':names,'tree':serialize_tree(model),
            'detector_config':cache['environment']['config'],'weights_hash':checkpoint_hash,
            'training_code_hashes':{p.name:sha(p) for p in Path(__file__).parent.glob('*.py')},
            'ablation':ablation,'gain_margin':margin,'validation_cost_weight':val_weight,
            'brightness_tree':serialize_tree(brightness_best[1]),'brightness_tree_selection':brightness_best[2],
            'brightness_tree_candidates':brightness_candidates,'fixed_candidates':fixed,
            'best_fixed_route':max(fixed,key=lambda r:r['val_objective'])['route'],
            'selection':chosen,'candidates':candidates,'pooled_relative_cost':cost,'diagnostics':stats,
            'brightness_threshold':best_threshold,'brightness_candidates':threshold_rows,
            'training_hashes':[r['sha256'] for r in tr], 'validation_hashes':[r['sha256'] for r in va],
            'profile_device':cache['environment']['config']['device'],
            'validation_objective':f'mean per-image F1 - {val_weight} * pooled relative extra route cost',
            'status':'Trained on supplied development data. Test quality and latency must be measured separately.'}
    write_json(out,policy)
    Path(out).with_suffix('.rules.txt').write_text(export_text(model,feature_names=list(names)),encoding='utf-8')
    print(json.dumps(stats,indent=2))

def evaluate(manifest,cfg,policy_path,out,split='test',repeats=3,limit=None):
    from .detector import Detector
    import psutil
    obj=read_json(manifest); audit(obj); rows=records_for(obj,split,limit)
    policy=read_json(policy_path); det=Detector(cfg); enh=Enhancer(cfg)
    if policy['training_manifest_hash']!=sha(manifest): raise ValueError('Manifest changed since training; restore it or rebuild development artifacts')
    if policy['signature']!=signature(cfg,det.weight_hash): raise ValueError('Policy/config/checkpoint mismatch; rebuild cache and policy')
    if split=='test' and {r['sha256'] for r in rows}&set(policy['training_hashes']+policy['validation_hashes']):
        raise ValueError('Held-out test overlaps development content')
    out=Path(out); out.mkdir(parents=True,exist_ok=False)
    cv2.setNumThreads(cfg['threads'])
    try: process=psutil.Process()
    except (psutil.Error, OSError): process=None
    warmup(det,enh,load_image(Path(obj['root'])/rows[0]['path']))
    methods=(*ROUTES,'threshold','selector')+tuple(a for a,key in [('brightness_tree','brightness_tree'),('best_fixed','best_fixed_route')] if key in policy); logs=[]; results={}
    rng=np.random.default_rng(cfg['seed'])
    for rep in range(repeats):
        for method in rng.permutation(methods):
            method=str(method); predictions={}; peak_rss=None
            if cfg['device']!='cpu': det.torch.cuda.reset_peak_memory_stats()
            for r in rows:
                det.sync(); begin=time.perf_counter()
                im=load_image(Path(obj['root'])/r['path']); decoded=time.perf_counter()
                route=select(im,method,cfg,policy); chosen=time.perf_counter()
                im=enh.apply(im,route); enhanced=time.perf_counter()
                pred=det.predict(im); det.sync(); end=time.perf_counter()
                if process is not None:
                    try: peak_rss=max(peak_rss or 0,process.memory_info().rss)
                    except (psutil.Error, OSError): process=None
                predictions[r['id']]=pred
                logs.append({'repeat':rep,'method':method,'id':r['id'],'route':route,
                  'decode_ms':(decoded-begin)*1000,'selector_ms':(chosen-decoded)*1000,
                  'enhance_ms':(enhanced-chosen)*1000,'detect_ms':(end-enhanced)*1000,'total_ms':(end-begin)*1000})
            key=f'{method}_repeat{rep}'
            write_json(out/f'{key}_predictions.json',predictions)
            m=summary(rows,predictions,cfg['operating_confidence'])
            timings=[x['total_ms'] for x in logs if x['repeat']==rep and x['method']==method]
            m.update({'mean_ms':float(np.mean(timings)),'p95_ms':float(np.percentile(timings,95)),
                      'fps':1000/float(np.mean(timings)),'rss_peak_sampled_mb':peak_rss/2**20 if peak_rss is not None else None,
                      'cuda_peak_allocated_mb':det.torch.cuda.max_memory_allocated()/2**20 if cfg['device']!='cpu' else None})
            results[key]=m
    with (out/'timings.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(logs[0])); writer.writeheader(); writer.writerows(logs)
    write_json(out/'summary.json',{'results':results,'environment':environment(cfg),'manifest_hash':sha(manifest),
       'policy_hash':sha(policy_path),'weights_hash':det.weight_hash,'split':split,'pilot_limit':limit,
       'timing_scope':'serial decoded-image pipeline including image decode, selection, enhancement, detector preprocessing, transfers, inference and postprocessing; excludes startup, metric computation, output serialization, camera acquisition and rendering',
       'selector_profile_device':policy['profile_device']})
    make_plot(out)

def make_plot(out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    d=read_json(Path(out)/'summary.json')['results']; fig,ax=plt.subplots(figsize=(8,5))
    for method in dict.fromkeys(k.rsplit('_repeat',1)[0] for k in d):
        v=[v for k,v in d.items() if k.startswith(method+'_repeat')]
        x=np.mean([r['mean_ms'] for r in v]); y=np.mean([r['map_50_95'] for r in v])*100
        ax.scatter(x,y); ax.annotate(method,(x,y),xytext=(4,4),textcoords='offset points')
    ax.set(xlabel='Mean decoded-image pipeline time (ms)',ylabel='COCO-style mAP 50:95 (%)',title='Measured quality and cost')
    ax.grid(alpha=.2); fig.tight_layout(); fig.savefig(Path(out)/'quality_latency.png',dpi=180); plt.close(fig)

def demo(cfg,policy_path,source,out):
    from .detector import Detector
    policy=read_json(policy_path); det=Detector(cfg); enh=Enhancer(cfg)
    if policy['signature']!=signature(cfg,det.weight_hash): raise ValueError('Policy/config/checkpoint mismatch')
    cap=cv2.VideoCapture(int(source) if str(source).isdigit() else str(source))
    if not cap.isOpened(): raise ValueError('Cannot open video/camera')
    writer=None; count=0
    try:
        while True:
            ok,im=cap.read()
            if not ok: break
            route=select(im,'selector',cfg,policy); processed=enh.apply(im,route); pred=det.predict(processed)
            for c,s,x1,y1,x2,y2 in pred:
                if s<cfg['operating_confidence']: continue
                from .core import NAMES
                cv2.rectangle(processed,(int(x1),int(y1)),(int(x2),int(y2)),(0,255,0),2)
                cv2.putText(processed,f'{NAMES[c]} {s:.2f}',(int(x1),max(18,int(y1))),0,.5,(0,255,0),1)
            cv2.putText(processed,f'Route: {route} | DEMO, not a timing benchmark',(10,25),0,.6,(0,255,255),2)
            if writer is None:
                fps=cap.get(cv2.CAP_PROP_FPS); fps=fps if np.isfinite(fps) and fps>0 else 25
                Path(out).parent.mkdir(parents=True,exist_ok=True)
                writer=cv2.VideoWriter(str(out),cv2.VideoWriter_fourcc(*'mp4v'),fps,(im.shape[1],im.shape[0]))
                if not writer.isOpened(): raise RuntimeError('Video writer failed')
            writer.write(processed); count+=1
    finally:
        cap.release()
        if writer: writer.release()
    print(f'Wrote {count} frames; playback FPS is source FPS, not measured inference throughput.')
