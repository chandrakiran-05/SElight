"""Publish frozen external experiments into the manuscript and local showcase."""
import argparse
import contextlib
import csv
import io
from pathlib import Path

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from selight.core import read_json, write_json, sha, load_image, Enhancer, select
from selight.external import ZeroDCEPP
from selight.metrics import quality
from paired_ap_comparison import evaluator, resampled_ap
from scripts.render_research_documents import NAMES, table, main as render


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--resamples', type=int, default=500)
    a = p.parse_args()
    out = Path('research'); datasets = {}
    for key, folder in [('LLVIP visible test pilot','llvip_visible_200_serial'),('Penn-Fudan daylight control','penn_fudan_170_serial')]:
        run = out/'experiments'/folder
        result = read_json(run/'summary.json')
        rows = read_json(run/'sample.json')['records']
        preds = {m:read_json(run/f'{m}_repeat0_predictions.json') for m in NAMES}
        be = evaluator(rows,preds['original'])
        pairs = {}
        for method in list(NAMES)[1:]:
            if preds[method] == preds['original']:
                interval = [0.,0.]
            else:
                ce = evaluator(rows,preds[method]); rng = np.random.default_rng(42); deltas = []
                for _ in range(a.resamples):
                    indices = rng.integers(0,len(rows),len(rows))
                    deltas.append(resampled_ap(ce,indices)-resampled_ap(be,indices))
                interval = np.percentile(deltas,[2.5,97.5]).tolist()
            pairs[method] = dict(ap_delta=result['aggregate'][method]['map_50_95']-result['aggregate']['original']['map_50_95'],image_bootstrap_95=interval,resamples=a.resamples,seed=42,scope='Conditional paired image bootstrap; scene independence unverified; no multiple-comparison correction')
        datasets[key] = dict(summary=result,paired=pairs,run=str(run).replace('\\','/'),summary_sha256=sha(run/'summary.json'))
        print('External intervals complete:',key,flush=True)
    write_json(out/'external_results.json',datasets)
    records = [[key,str(v['summary']['dataset_audit']['images']),NAMES[m],f'{s["map_50_95"]*100:.3f}',f'{s["map_50"]*100:.3f}',f'{s["f1"]:.4f}',f'{s["mean_ms"]:.2f}',f'{s["p95_ms"]:.2f}'] for key,v in datasets.items() for m,s in v['summary']['aggregate'].items()]
    headers = ['Dataset','Images','Method','Person AP50:95 %','Person AP50 %','Micro F1','Mean ms','P95 ms']
    with (out/'external_comparison.csv').open('w',newline='',encoding='utf-8') as f:
        w = csv.writer(f); w.writerow(headers); w.writerows(records)
    ci = [[key,NAMES[m],f'{z["ap_delta"]*100:+.3f}','['+', '.join(f'{x*100:+.3f}' for x in z['image_bootstrap_95'])+']'] for key,v in datasets.items() for m,z in v['paired'].items()]
    section = '''### 5.6 Frozen external dataset transfer

We import 200 of the 3,463 visible test images from the LLVIP archive mirrored on Hugging Face [11], pinned at revision e50976add2fcca00b0db0f3bba59ed486d8c1f6d. The cohort consists of the first 200 visible/test archive paths sorted by SHA256(path), declared before inference; it is a sampled pilot, not the full official benchmark. Original-resolution visible images are used; infrared and training images are not imported. LLVIP is predominantly low-light, but contains some daytime scenes, so we do not assert that every sampled frame is night. The mirror's equivalence to the official 2023 corrected annotation release is unverified. We interpret its XML as PASCAL VOC 1-based inclusive endpoints and convert to 0-based half-open edges; the mirror does not explicitly resolve the original coordinate convention.

We also import all 170 official Penn-Fudan images and their human instance masks from the University of Pennsylvania website [12]. Tight person boxes use minimum foreground coordinates and maximum coordinates plus one. The released archive contains 423 person instances: 345 described on the homepage plus 78 small/occluded additions documented in its readme.txt and added-object-list.txt. Both mask IDs and supplied text-annotation counts verify 423. This is a daylight domain control, not evidence of low-light performance, and no official held-out split is claimed.

The ExDark-trained policy, YOLOv8n checkpoint, enhancer settings and thresholds are frozen. No fitting or parameter selection uses either external cohort. Only person detections are scored because other detector classes are unannotated; person AP is not comparable to five-class ExDark AP. Exact file hashes find no within-cohort duplicate or overlap with the reference ExDark manifest. Unknown overlap with detector pretraining, near duplicates and scene/session dependence remain limitations. The two datasets run sequentially on the same GPU, each with ten warmups per method, three repetitions and randomized method/image order. Timing excludes image decode, metrics and rendering. Earlier overlapping runs are excluded from timing and final results.

'''+table(headers,records)+'''

*Table 6. Fresh external person-only detection results; accuracy uses repetition zero and timing pools three repetitions. These are local measurements, not published benchmark scores.*

'''+table(['Dataset','Candidate','AP delta (pp)','Conditional 95% interval (pp)'],ci)+f'''

*Table 7. Paired person AP differences relative to original input with {a.resamples} image resamples (seed 42). Intervals are conditional on images; LLVIP frames from common scenes may be correlated, and intervals do not establish population-level significance. Identical selector predictions have an exact zero difference.*

Both external cohorts retain exact SELight/original prediction equality. Enhancement point estimates are lower than original person AP on these cohorts. The results extend the negative selector finding across two external image collections without establishing five-class or session-independent generalization. `external_results.json`, `external_comparison.csv` and the local showcase retain provenance, raw predictions, route counts, timings and diagnostic examples.

![External dataset comparison](figures/external_comparison.png)

*Figure 6. Person AP50:95 for the two separately evaluated external cohorts. LLVIP is a visible test pilot; Penn-Fudan is a daylight control. The collections are not pooled into one accuracy value.*

'''
    paper_path = out/'PAPER.md'; paper = paper_path.read_text(encoding='utf-8')
    if '### 5.6 Frozen external dataset transfer' in paper:
        paper = paper[:paper.index('### 5.6 Frozen external dataset transfer')]+paper[paper.index('## 6. Discussion and limitations'):]
    paper = paper.replace('## 6. Discussion and limitations',section+'## 6. Discussion and limitations')
    if '11. Xinyu Jia' not in paper:
        paper += '\n11. Xinyu Jia et al. *LLVIP: A Visible-infrared Paired Dataset for Low-light Vision.* ICCV Workshops 2021. [Paper](https://arxiv.org/abs/2108.10831), [official dataset and terms](https://github.com/bupt-ai-cz/LLVIP), [pinned Hugging Face mirror](https://huggingface.co/datasets/McCheng/LLVIP/tree/e50976add2fcca00b0db0f3bba59ed486d8c1f6d).\n12. Liming Wang, Jianbo Shi, Gang Song and I-fan Shen. *Object Detection Combining Recognition and Segmentation.* ACCV 2007. [Official Penn-Fudan dataset](https://www.cis.upenn.edu/~jshi/ped_html/).\n'
    abstract_extra = ' Frozen external person-only evaluation on 200 LLVIP visible test images and all 170 Penn-Fudan images also produces exact selector/original equality; Penn-Fudan is a daylight control.'
    if abstract_extra not in paper: paper=paper.replace('**Keywords:**',abstract_extra+'\n\n**Keywords:**')
    paper_path.write_text(paper,encoding='utf-8')
    evidence = read_json(out/'evidence.json'); evidence['external_transfer'] = datasets; write_json(out/'evidence.json',evidence)
    fig,axes = plt.subplots(1,2,figsize=(11,4))
    for ax,(key,v) in zip(axes,datasets.items()):
        values=[v['summary']['aggregate'][m]['map_50_95']*100 for m in NAMES]
        ax.bar(range(5),values,color=['#36596b','#829aaa','#9fbba9','#236d66','#c69b61'])
        ax.set_xticks(range(5),['Original','Gamma','CLAHE','SELight','Zero-DCE++'],rotation=20); ax.set_ylabel('Person AP50:95 (%)'); ax.set_title(key); ax.set_ylim(0,100)
        for i,x in enumerate(values): ax.text(i,x+1,f'{x:.2f}',ha='center',fontsize=9)
    fig.tight_layout(); fig.savefig(out/'figures/external_comparison.png',dpi=180); fig.savefig(out/'figures/external_comparison.pdf'); plt.close(fig)
    cfg=read_json('configs/gpu.json'); enh=Enhancer(cfg); ext=ZeroDCEPP(cfg['device']); policy=read_json('models/policy.json'); gallery=[]
    for key,v in datasets.items():
        run=Path(v['run']); rows=read_json(run/'sample.json')['records']; root=Path(v['summary']['source_manifest']['root'])
        preds={m:read_json(run/f'{m}_repeat0_predictions.json') for m in NAMES}
        chosen=[min(rows,key=lambda r:quality(r['boxes'],preds['original'][r['id']])), max(rows,key=lambda r:quality(r['boxes'],preds[ext.name][r['id']])-quality(r['boxes'],preds['original'][r['id']]))]
        for case,r in enumerate(chosen):
            image=load_image(root/r['path']); panels=[]
            for m in NAMES:
                processed=ext.apply(image) if m==ext.name else enh.apply(image,select(image,m,cfg,policy))
                canvas=processed.copy()
                for _,x1,y1,x2,y2 in r['boxes']: cv2.rectangle(canvas,(int(x1),int(y1)),(int(x2),int(y2)),(0,230,255),3)
                for _,s,x1,y1,x2,y2 in preds[m][r['id']]:
                    if s>=.25: cv2.rectangle(canvas,(int(x1),int(y1)),(int(x2),int(y2)),(255,230,0),2)
                canvas=cv2.resize(canvas,(300,240)); cv2.putText(canvas,NAMES[m],(8,22),cv2.FONT_HERSHEY_SIMPLEX,.48,(255,255,255),2); panels.append(canvas)
            name=f'external_{r["id"].replace("/","_")}_{case}.png'; cv2.imwrite(str(out/'figures'/name),np.concatenate(panels,axis=1))
            gallery.append(dict(dataset=key,id=r['id'],file='figures/'+name,selection='Lowest original F1' if case==0 else 'Largest Zero-DCE++ minus original F1 (may still be nonpositive)'))
    write_json(out/'external_gallery.json',gallery)
    # Refresh main document outputs, then add external results to the standalone HTML.
    import sys
    sys.argv=['render_research_documents']; render()
    page=(out/'showcase.html').read_text(encoding='utf-8')
    htmlrows=''.join('<tr>'+''.join('<td>'+x+'</td>' for x in r)+'</tr>' for r in [headers,*records])
    cases=''.join('<article><h3>'+g['dataset']+' / '+g['id']+'</h3><p>'+g['selection']+'; GT yellow, person predictions cyan at confidence 0.25.</p><img style="width:100%" src="'+g['file']+'"></article>' for g in gallery)
    extra='<section><h2>External human-labelled datasets</h2><p>200 LLVIP visible test images from Hugging Face; 170 official Penn-Fudan daylight images. Frozen model, person-only scoring, sequential GPU runs. Sampled LLVIP pilot; no retraining or five-class generalization claim.</p><img style="width:100%" src="figures/external_comparison.png"><div class="scroll" style="overflow-x:auto"><table>'+htmlrows+'</table></div><p><a href="external_results.json">Provenance and intervals</a> · <a href="external_comparison.csv">External results CSV</a></p>'+cases+'</section>'
    page=page.replace('<footer>',extra+'<footer>'); (out/'showcase.html').write_text(page,encoding='utf-8')
    write_json(out/'artifact_hashes.json',{str(x.relative_to(out)).replace('\\','/'):sha(x) for x in out.rglob('*') if x.is_file() and x.name!='artifact_hashes.json'})


if __name__ == '__main__':
    main()
