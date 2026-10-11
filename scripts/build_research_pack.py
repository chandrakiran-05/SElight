"""Recompute stored metrics, paired intervals, figures and an auditable code map.

Run AFTER scripts.research_compare. Historical predictions are re-evaluated,
not described as new inference. No training, test inference or source mutation.
"""
import argparse
import ast
import collections
import contextlib
import csv
import io
from pathlib import Path

import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch
from selight.core import NAMES, ROUTES, Enhancer, features, load_image, read_json, select, sha, write_json
from selight.experiment import route_label, records_for
from selight.metrics import quality, summary
from selight.external import ZeroDCEPP
from paired_ap_comparison import evaluator, resampled_ap

LABELS = {'original':'YOLOv8n original', 'gamma':'Gamma + YOLOv8n',
          'clahe':'CLAHE + YOLOv8n', 'selector':'SELight selector',
          'zero_dcepp_scale12_pad':'Zero-DCE++ + YOLOv8n'}
COLORS = {'original':'#355070', 'gamma':'#e9a23b', 'clahe':'#47856a',
          'selector':'#8057a6', 'zero_dcepp_scale12_pad':'#cd6262'}


def measured(rows, pred):
    with contextlib.redirect_stdout(io.StringIO()):
        return summary(rows, pred)


def figure_save(fig, path):
    fig.savefig(path.with_suffix('.png'), dpi=190, bbox_inches='tight', facecolor='white')
    fig.savefig(path.with_suffix('.pdf'), bbox_inches='tight', facecolor='white')
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--backup', required=True)
    p.add_argument('--out', default='research')
    p.add_argument('--resamples', type=int, default=1000)
    a = p.parse_args()
    if a.resamples < 1: p.error('resamples must be positive')
    root, backup, out = Path.cwd(), Path(a.backup).resolve(), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    figs = out/'figures'; figs.mkdir(exist_ok=True)
    sources = {}

    def record(path):
        sources[str(path.resolve())] = sha(path)
        return read_json(path)

    manifest_path = backup/'data/manifest.json'
    manifest = record(manifest_path)
    reviewed = record(backup/'data/manifest_reviewed.json')
    policy = record(root/'models/policy.json')
    cfg = record(root/'configs/gpu.json')
    fresh_dir = out/'experiments/validation_200'
    fresh = record(fresh_dir/'summary.json')
    rows = record(fresh_dir/'sample.json')['records']
    if fresh['manifest_hash'] != sha(manifest_path) or fresh['policy_hash'] != sha(root/'models/policy.json'):
        raise ValueError('Fresh comparison provenance mismatch')
    predictions = {m:record(fresh_dir/f'{m}_repeat0_predictions.json') for m in LABELS}
    checks = []
    for m,pred in predictions.items():
        if set(pred) != {r['id'] for r in rows}: raise ValueError('Comparison IDs differ')
        new = measured(rows,pred)
        delta = new['map_50_95']-fresh['aggregate'][m]['map_50_95']
        if abs(delta)>1e-12: raise ValueError('Fresh summary does not reproduce')
        checks.append(dict(source='fresh',method=m,map_recompute_error=delta))

    legacy = {}
    test_rows = records_for(manifest, 'test')
    for device in ['gpu','cpu']:
        run = backup/f'runs/test_{device}'
        saved = record(run/'summary.json')
        if saved['manifest_hash'] != sha(manifest_path): raise ValueError('Legacy manifest mismatch')
        timings_path = run/'timings.csv'; sources[str(timings_path.resolve())]=sha(timings_path)
        with timings_path.open(newline='',encoding='utf-8') as f: logs = list(csv.DictReader(f))
        legacy[device] = {}
        original = None
        for m in ['original','gamma','clahe','threshold','selector']:
            pred = record(run/f'{m}_repeat0_predictions.json')
            if set(pred)!={r['id'] for r in test_rows}: raise ValueError('Legacy IDs mismatch')
            metric = measured(test_rows,pred)
            error = metric['map_50_95']-saved['results'][m+'_repeat0']['map_50_95']
            if abs(error)>1e-10: raise ValueError('Historical metric does not reproduce')
            entries = [x for x in logs if x['method']==m]
            times = np.array([float(x['total_ms']) for x in entries])
            metric.update(mean_ms=float(times.mean()),p95_ms=float(np.percentile(times,95)),
                          median_ms=float(np.median(times)),maximum_ms=float(times.max()),
                          timing_scope=saved['timing_scope'], provenance='Stored test predictions; metrics recomputed now; timings inherited')
            if m=='original': original=pred
            if m=='selector': metric['predictions_equal_original']=pred==original
            legacy[device][m]=metric
            checks.append(dict(source='legacy_'+device,method=m,map_recompute_error=error))
        worst=max(logs,key=lambda x:float(x['total_ms']))
        legacy[device]['largest_timing_event'] = worst
        print('Recomputed all historical '+device+' test metrics',flush=True)

    validation = record(root/'outputs/ablations/summary.json')
    val_rows = records_for(manifest, 'val')
    for m in validation['results']:
        pred=record(root/f'outputs/ablations/{m}_val_predictions.json')
        new=measured(val_rows,pred)
        if abs(new['map_50_95']-validation['results'][m]['map_50_95'])>1e-12:
            raise ValueError('Ablation does not reproduce: '+m)
    cache = record(backup/'runs/dev_cache.json')
    if cache['manifest_hash']!=sha(manifest_path): raise ValueError('Development cache provenance mismatch')
    train=[r for r in cache['records'] if r['split']=='train']
    tr_hash={r['sha256'] for r in train}; va_hash={r['sha256'] for r in cache['records'] if r['split']=='val'}
    if tr_hash & va_hash: raise ValueError('Exact development overlap')
    if {r['sha256'] for r in test_rows} & (tr_hash|va_hash): raise ValueError('Exact test/development overlap')
    label_counts=collections.Counter(route_label(r['quality'],policy['pooled_relative_cost'],policy['selection']['training_lambda'],policy['gain_margin']) for r in train)
    near_omitted={r['id'] for r in manifest['records']}-{r['id'] for r in reviewed['records']}
    fresh_near_overlap=sorted(near_omitted & {r['id'] for r in rows})
    by_split = {s:dict(images=sum(r['split']==s for r in manifest['records']),
                      target_negative_images=sum(r['split']==s and not r['boxes'] for r in manifest['records'])) for s in ['train','val','test']}
    positive=[r for r in rows if r['boxes']]; negative=[r for r in rows if not r['boxes']]
    base_q=np.array([quality(r['boxes'],predictions['original'][r['id']]) for r in rows])
    paired={}
    be=evaluator(rows,predictions['original'])
    rng=np.random.default_rng(42)
    indices=[rng.integers(0,len(rows),len(rows)) for _ in range(a.resamples)]
    for m in ['gamma','clahe','selector','zero_dcepp_scale12_pad']:
        q=np.array([quality(r['boxes'],predictions[m][r['id']]) for r in rows]); diff=q-base_q
        f1_boot=[float(diff[i].mean()) for i in indices]
        if predictions[m]==predictions['original']:
            ap_delta=0.; ci=[0.,0.]; valid=a.resamples
            ap_note='Analytical zero interval because predictions exactly equal; resampling unnecessary'
        else:
            ce=evaluator(rows,predictions[m]); ap_delta=fresh['aggregate'][m]['map_50_95']-fresh['aggregate']['original']['map_50_95']
            bootstrap=[]
            for i in indices:
                bp,cp=resampled_ap(be,i),resampled_ap(ce,i)
                if bp is not None and cp is not None: bootstrap.append(cp-bp)
            ci=np.percentile(bootstrap,[2.5,97.5]).tolist(); valid=len(bootstrap)
            ap_note='Paired image resampling; dataset-level COCO AP recomputed, original annotation area'
        paired[m]=dict(ap_difference=ap_delta,ap_percentile_95_interval=ci,
            mean_image_f1_difference=float(diff.mean()),f1_percentile_95_interval=np.percentile(f1_boot,[2.5,97.5]).tolist(),
            improved_images=int((diff>0).sum()),degraded_images=int((diff<0).sum()),equal_images=int((diff==0).sum()),
            resamples=a.resamples,valid_ap_resamples=valid,seed=42,images=len(rows),independence='Images assumed independent; capture sessions unavailable',ap_note=ap_note)
        print('Paired intervals: '+m+' '+str(ci),flush=True)
    strata={m:dict(target_positive_mean_image_f1=float(np.mean([quality(r['boxes'],pred[r['id']]) for r in positive])),
                   target_negative_mean_image_f1=float(np.mean([quality(r['boxes'],pred[r['id']]) for r in negative])),
                   target_positive_images=len(positive),target_negative_images=len(negative)) for m,pred in predictions.items()}
    stress=record(backup/'runs/stress_val_pilot/summary.json')
    sensitivity=record(backup/'runs/reviewed_split/summary.json')
    evidence=dict(fresh=fresh,paired=paired,legacy=legacy,validation_ablations=validation,
        reviewed_split_sensitivity=sensitivity,dataset_splits=by_split,
        train_route_labels=dict(label_counts),leaf_routes=[policy['tree']['route'][i] for i,l in enumerate(policy['tree']['left']) if l==-1],
        diagnostics=policy['diagnostics'],stratified_fresh_f1=strata,
        fresh_near_duplicate_omission_ids=fresh_near_overlap,
        checks=checks,source_hashes=sources,
        date='2026-10-12',new_test_inference=False,new_training=False)
    write_json(out/'evidence.json',evidence)
    csv_rows=[]
    for scope,device,methods in [('fresh_validation_200','gpu',fresh['aggregate']),('legacy_test_2560','gpu',legacy['gpu']),('legacy_test_2560','cpu',legacy['cpu'])]:
        for m,v in methods.items():
            if 'map_50_95' not in v: continue
            csv_rows.append(dict(scope=scope,device=device,method=m,images=v['images'],map_50_95=v['map_50_95'],map_50=v['map_50'],micro_f1=v['f1'],mean_image_f1=v['mean_image_f1'],mean_ms=v['mean_ms'],p95_ms=v['p95_ms'],timing_scope=fresh['timing_scope'] if scope.startswith('fresh') else v['timing_scope']))
    with (out/'comparison.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(csv_rows[0]));w.writeheader();w.writerows(csv_rows)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    methods=list(LABELS); agg=fresh['aggregate']
    fig,axes=plt.subplots(1,2,figsize=(12,4.5))
    names=['Original','Gamma','CLAHE','SELight','Zero-DCE++']
    for ax,key,title in zip(axes,['map_50_95','mean_ms'],['COCO-style AP50:95 (%)','Mean processing time (ms)']):
        values=[agg[m][key]*(100 if key=='map_50_95' else 1) for m in methods]
        bars=ax.bar(names,values,color=[COLORS[m] for m in methods]);ax.set_ylabel(title);ax.set_ylim(0,max(values)*1.22)
        for bar,v in zip(bars,values):ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+max(values)*.025,f'{v:.2f}',ha='center',fontsize=9)
        ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    fig.suptitle('Matched 200-image validation comparison | frozen YOLOv8n | RTX 4050')
    fig.text(.5,-.02,'Accuracy from repeat 0. Three timing repeats; decode/acquisition/rendering excluded. Zero-DCE++ uses padding adaptation.',ha='center',fontsize=9)
    fig.tight_layout();figure_save(fig,figs/'fresh_comparison')
    fig,ax=plt.subplots(figsize=(9,4))
    candidates=['gamma','clahe','selector','zero_dcepp_scale12_pad']
    delta=np.array([paired[m]['ap_difference']*100 for m in candidates]);ci=np.array([paired[m]['ap_percentile_95_interval'] for m in candidates])*100
    ax.errorbar(delta,np.arange(4),xerr=np.maximum(0,np.vstack([delta-ci[:,0],ci[:,1]-delta])),fmt='o',color='#355070',capsize=4)
    ax.set_yticks(np.arange(4),[LABELS[m] for m in candidates]);ax.axvline(0,color='#888',ls='--');ax.set_xlabel('AP50:95 difference from original (percentage points)');ax.set_title(f'Paired image bootstrap | {a.resamples:,} resamples | 200 validation images');ax.grid(axis='x',alpha=.2)
    fig.tight_layout();figure_save(fig,figs/'paired_intervals')
    fig,ax=plt.subplots(figsize=(10,4));x=np.arange(5);width=.25
    for j,m in enumerate(['original','gamma','clahe']):
        ax.bar(x+(j-1)*width,[legacy['gpu'][m]['per_class_ap'][n]*100 for n in NAMES],width,label=LABELS[m],color=COLORS[m])
    ax.set_xticks(x,NAMES);ax.set_ylabel('AP50:95 (%)');ax.legend(frameon=False,ncol=3,fontsize=8);ax.set_title('Stored GPU test predictions, recomputed | 2,560 images');ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    fig.tight_layout();figure_save(fig,figs/'legacy_per_class')
    fig,ax=plt.subplots(figsize=(9,4));conditions=list(stress['conditions'])
    for m in ['original','gamma','clahe']:
        vals=[stress['results'][f'{c}__{m}__repeat0']['map_50_95']*100 for c in conditions]
        ax.plot(range(4),vals,'o-',label=LABELS[m],color=COLORS[m])
    ax.set_xticks(range(4),['Native','Max-side 640','Max-side 320','320 + noise + JPEG']);ax.set_ylabel('AP50:95 (%)');ax.set_title('Historical 30-image validation stress pilot | synthetic inputs');ax.legend(frameon=False);ax.grid(alpha=.2)
    fig.tight_layout();figure_save(fig,figs/'stress_pilot')
    fig,ax=plt.subplots(figsize=(11,3.2));ax.axis('off')
    for x,text in [(.02,'RGB image\nBGR uint8'),(.22,'64 x 64 features\n8 statistics'),(.44,'Frozen CART\nchoose one route'),(.66,'Original / gamma\n/ Lab CLAHE'),(.85,'Frozen YOLOv8n\n5-class boxes')]:
        box=FancyBboxPatch((x,.38),.13,.34,boxstyle='round,pad=0.02',transform=ax.transAxes,fc='#f0f3f6',ec='#355070');ax.add_patch(box);ax.text(x+.065,.55,text,transform=ax.transAxes,ha='center',va='center',fontsize=9)
    for x in [.18,.40,.62,.81]:ax.annotate('',xy=(x+.025,.55),xytext=(x-.025,.55),xycoords=ax.transAxes,arrowprops={'arrowstyle':'->','color':'#355070'})
    ax.text(.5,.14,'Observed saved policy: all leaves choose original. Exactly one detector call per image.',transform=ax.transAxes,ha='center',fontsize=11)
    figure_save(fig,figs/'pipeline')
    inventory=[]
    paths=sorted(root.glob('*.py'))+sorted((root/'selight').glob('*.py'))+sorted((root/'scripts').glob('*.py'))+sorted((root/'tests').glob('*.py'))
    for path in paths:
        text=path.read_text(encoding='utf-8-sig');t=ast.parse(text)
        functions=[dict(name=n.name,line=n.lineno,kind=type(n).__name__,doc=ast.get_docstring(n)) for n in t.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))]
        imports=sorted({n.module or '' for n in ast.walk(t) if isinstance(n,ast.ImportFrom)}|{z.name for n in ast.walk(t) if isinstance(n,ast.Import) for z in n.names})
        inventory.append(dict(path=str(path.relative_to(root)).replace('\\','/'),lines=len(text.splitlines()),sha256=sha(path),definitions=functions,imports=imports,module_doc=ast.get_docstring(t)))
    write_json(out/'code_inventory.json',dict(files=inventory,total_python_lines=sum(x['lines'] for x in inventory),scope='Owned Python source and tests, excluding environment and third-party code'))
    # Gallery cases are selected by declared utility rules, not claimed representative.
    gain=np.array([quality(r['boxes'],predictions['zero_dcepp_scale12_pad'][r['id']])-quality(r['boxes'],predictions['original'][r['id']]) for r in rows])
    chosen=[];used=set()
    def add(index,reason):
        if index not in used:chosen.append((index,reason));used.add(index)
    add(int(np.argmax(gain)),'Largest Zero-DCE++ per-image F1 gain')
    add(int(np.argmin(gain)),'Largest Zero-DCE++ per-image F1 decline')
    pos_indices=[i for i,r in enumerate(rows) if r['boxes']]
    add(min(pos_indices,key=lambda i:(base_q[i],rows[i]['id'])),'Lowest original F1 among target-positive images')
    for c in range(5):
        match=[i for i,r in enumerate(rows) if any(int(b[0])==c for b in r['boxes'])]
        if match:add(match[0],'First hash-sampled example with '+NAMES[c])
    ext=ZeroDCEPP(cfg['device']);enh=Enhancer(cfg)
    gallery=[]
    for number,(index,reason) in enumerate(chosen):
        r=rows[index];image=load_image(Path(manifest['root'])/r['path'])
        variants=[('Ground truth',image,None)]+[(LABELS[m],ext.apply(image) if m=='zero_dcepp_scale12_pad' else enh.apply(image,select(image,m,cfg,policy)),m) for m in methods]
        fig,axes=plt.subplots(1,6,figsize=(19,4.3))
        for ax,(label,img,m) in zip(axes,variants):
            ax.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB));ax.axis('off')
            for c,x1,y1,x2,y2 in r['boxes']:
                ax.add_patch(Rectangle((x1,y1),x2-x1,y2-y1,fill=False,ec='#ffe45c',lw=1.5,ls='--'))
                if m is None:ax.text(x1,y1,NAMES[int(c)],color='#ffe45c',fontsize=7,bbox=dict(fc='black',alpha=.7,pad=1))
            if m is not None:
                for c,s,x1,y1,x2,y2 in predictions[m][r['id']]:
                    if s<cfg['operating_confidence']:continue
                    ax.add_patch(Rectangle((x1,y1),x2-x1,y2-y1,fill=False,ec='#37e0ee',lw=1.5))
                    ax.text(x1,y1,f'{NAMES[int(c)]} {s:.2f}',color='#37e0ee',fontsize=7,bbox=dict(fc='black',alpha=.7,pad=1))
            suffix='' if m is None else f'\nF1={quality(r["boxes"],predictions[m][r["id"]]):.3f}'
            ax.set_title(label+suffix,fontsize=10)
        fig.suptitle(r['id']+' | '+reason+' | yellow dashed: GT; cyan solid: predictions >= .25',fontsize=12)
        fig.tight_layout();name=f'case_{number+1:02d}.png';fig.savefig(figs/name,dpi=150,bbox_inches='tight',facecolor='white');plt.close(fig)
        gallery.append(dict(id=r['id'],reason=reason,file='figures/'+name,ground_truth=r['boxes'],f1={m:quality(r['boxes'],pred[r['id']]) for m,pred in predictions.items()}))
    write_json(out/'gallery.json',gallery)
    # Sources gathered before rendering are immutable inputs; new assets have own manifest.
    write_json(out/'artifact_hashes.json',{str(x.relative_to(out)).replace('\\','/'):sha(x) for x in out.rglob('*') if x.is_file() and x.name!='artifact_hashes.json'})
    print('Evidence, source inventory, CSV, figures and gallery complete.',flush=True)


if __name__=='__main__':
    main()
