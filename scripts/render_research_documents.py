"""Render authored manuscript tables, standalone TeX and local HTML showcase.

The prose remains authored PAPER.md. Named table markers permit regeneration
against evidence.json. TeX plots use inline picture primitives, with no external
project files required by the built-in standalone compiler.
"""
import argparse
import ast
import html
import re
from pathlib import Path
from selight.core import read_json, write_json, sha

NAMES = {'original':'Original','gamma':'Gamma','clahe':'CLAHE','selector':'SELight','zero_dcepp_scale12_pad':'Zero-DCE++ (adapt.)'}


def table(headers, rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(map(str,r))+' |' for r in rows])


def tex_escape(s):
    mapping={'\\':r'\textbackslash{}','&':r'\&','%':r'\%','$':r'\$','#':r'\#','_':r'\_','{':r'\{','}':r'\}','~':r'\textasciitilde{}','^':r'\textasciicircum{}','—':'---','–':'--','’':"'",'“':'``','”':"''"}
    return ''.join(mapping.get(c,c) for c in s)


def inline(s):
    result=[];start=0
    pattern=r'\[([^\]]+)\]\(([^)]+)\)|`([^`]+)`|\*\*([^*]+)\*\*|\*([^*]+)\*'
    for m in re.finditer(pattern,s):
        result.append(tex_escape(s[start:m.start()]))
        if m.group(1):result.append(r'\href{'+tex_escape(m.group(2))+'}{'+tex_escape(m.group(1))+'}')
        elif m.group(3):result.append(r'\texttt{'+tex_escape(m.group(3))+'}')
        elif m.group(4):result.append(r'\textbf{'+tex_escape(m.group(4))+'}')
        else:result.append(r'\emph{'+tex_escape(m.group(5))+'}')
        start=m.end()
    result.append(tex_escape(s[start:]));return ''.join(result)


def picture(kind,evidence):
    agg=evidence['fresh']['aggregate'];methods=list(NAMES)
    commands=[]
    if kind=='external_comparison':
        commands=[r'\begin{picture}(440,155)']
        for panel,(name,data) in enumerate(evidence['external_transfer'].items()):
            start=panel*225
            title='LLVIP visible pilot' if panel==0 else 'Penn-Fudan daylight'
            commands.append(fr'\put({start+110},140){{\makebox(0,0){{\textbf{{{title}}}}}}}')
            for i,m in enumerate(methods):
                val=data['summary']['aggregate'][m]['map_50_95']*100;y=115-i*22;w=110*val/100
                label=NAMES[m].replace(' (adapt.)','')
                commands.append(fr'\put({start},{y}){{\makebox(0,0)[l]{{\scriptsize {label}}}}}')
                commands.append(fr'\put({start+69},{y-3}){{\color{{teal}}\rule{{{w:.2f}pt}}{{6pt}}}}')
                commands.append(fr'\put({start+72+w:.2f},{y}){{\makebox(0,0)[l]{{\scriptsize {val:.2f}}}}}')
        commands.append(r'\put(220,0){\makebox(0,0){Person AP50:95 (\%); different datasets, not pooled accuracy.}}')
    elif kind=='pipeline':
        commands=[r'\begin{picture}(440,90)']
        texts=['RGB input','64x64\\8 features','Frozen CART','One transform','Frozen YOLO\\5-class boxes']
        for i,text in enumerate(texts):
            x=i*90;commands.append(fr'\put({x},40){{\framebox(75,35){{\shortstack{{{text}}}}}}}')
            if i<4:commands.append(fr'\put({x+75},57){{\vector(1,0){{15}}}}')
        commands.append(r'\put(220,15){\makebox(0,0){All saved leaves select original; one detector call per image.}}')
    elif kind=='fresh_comparison':
        commands=[r'\begin{picture}(440,150)']
        for panel,(key,maxv,title) in enumerate([('map_50_95',50,'AP50:95 (%)'),('mean_ms',40,'Processing (ms)')]):
            start=panel*225
            commands.append(fr'\put({start+110},140){{\makebox(0,0){{\textbf{{{title}}}}}}}')
            for i,m in enumerate(methods):
                val=agg[m][key]*(100 if key=='map_50_95' else 1);y=115-i*22;w=110*val/maxv
                name=NAMES[m].replace(' (adapt.)','')
                commands.append(fr'\put({start}, {y}){{\makebox(0,0)[l]{{\scriptsize {name}}}}}')
                commands.append(fr'\put({start+69},{y-3}){{\color{{teal}}\rule{{{w:.2f}pt}}{{6pt}}}}')
                commands.append(fr'\put({start+72+w:.2f},{y}){{\makebox(0,0)[l]{{\scriptsize {val:.2f}}}}}')
    elif kind=='paired_intervals':
        commands=[r'\begin{picture}(440,150)',r'\put(260,25){\line(0,1){110}}']
        for i,m in enumerate(['gamma','clahe','selector','zero_dcepp_scale12_pad']):
            v=evidence['paired'][m];lo,hi=[100*x for x in v['ap_percentile_95_interval']];delta=v['ap_difference']*100;y=115-i*25
            left,right=260+lo*45,260+hi*45
            commands += [fr'\put(0,{y}){{\makebox(0,0)[l]{{\small {NAMES[m]}}}}}',fr'\put({left:.2f},{y}){{\line(1,0){{{max(0,right-left):.2f}}}}}',fr'\put({left:.2f},{y-3}){{\line(0,1){{6}}}}',fr'\put({right:.2f},{y-3}){{\line(0,1){{6}}}}',fr'\put({260+delta*45:.2f},{y}){{\circle*{{4}}}}']
        for tick in [-2,-1,0,1,2]:commands.append(fr'\put({260+tick*45},15){{\makebox(0,0){{{tick}}}}}')
        commands.append(r'\put(260,0){\makebox(0,0){\scriptsize AP difference (percentage points)}}')
    elif kind=='legacy_per_class':
        commands=[r'\begin{picture}(440,165)']
        classes=['person','bicycle','car','motorcycle','bus'];colors=['black','gray','teal']
        for c,name in enumerate(classes):
            x=45+c*80
            commands.append(fr'\put({x+15},20){{\makebox(0,0){{\scriptsize {name}}}}}')
            for j,m in enumerate(['original','gamma','clahe']):
                val=evidence['legacy']['gpu'][m]['per_class_ap'][name]*100
                commands.append(fr'\put({x+j*13},35){{\color{{{colors[j]}}}\rule{{11pt}}{{{val*1.7:.2f}pt}}}}')
        commands.append(r'\put(220,155){\makebox(0,0){Original (black), gamma (gray), CLAHE (teal); AP50:95 percent}}')
        commands.append(r'\put(0,35){\line(1,0){425}}')
        for t in [0,20,40,60]:commands.append(fr'\put(15,{35+t*1.7}){{\makebox(0,0){{\scriptsize {t}}}}}')
    elif kind=='stress_pilot':
        # Values are read from the saved, hash-bound historical stress artifact.
        backup=evidence['source_hashes']
        source=next(Path(p) for p in backup if p.replace('\\','/').endswith('runs/stress_val_pilot/summary.json'))
        s=read_json(source);conditions=list(s['conditions']);commands=[r'\begin{picture}(440,175)']
        for j,m in enumerate(['original','gamma','clahe']):
            coords=[(50+i*115,40+s['results'][f'{c}__{m}__repeat0']['map_50_95']*170) for i,c in enumerate(conditions)]
            commands.append('{'+r'\color{'+['black','gray','teal'][j]+'}')
            for x,y in coords:commands.append(fr'\put({x},{y:.2f}){{\circle*{{4}}}}')
            for (x1,y1),(x2,y2) in zip(coords,coords[1:]):commands.append(fr'\qbezier({x1},{y1:.2f})({(x1+x2)/2:.2f},{(y1+y2)/2:.2f})({x2},{y2:.2f})')
            commands.append('}')
        for i,label in enumerate(['Native','640','320','320+noise+JPEG']):commands.append(fr'\put({50+i*115},20){{\makebox(0,0){{\scriptsize {label}}}}}')
        commands.append(r'\put(220,155){\makebox(0,0){AP50:95 percent; original (black), gamma (gray), CLAHE (teal)}}')
        for t in [0,20,40,60]:commands.append(fr'\put(15,{40+t*1.7}){{\makebox(0,0){{\scriptsize {t}}}}}')
    commands.append(r'\end{picture}')
    return '\n'.join(commands)


def make_tex(markdown,evidence):
    lines=markdown.splitlines();body=[];i=0;references=False
    while i<len(lines):
        line=lines[i]
        if line.startswith('<!--'):i+=1;continue
        if line.startswith('# '):i+=1;continue
        if line.startswith('## '):
            title=re.sub(r'^\d+\.\s*','',line[3:]);body.append(r'\section{'+tex_escape(title)+'}')
            if title=='References':references=True;body.append(r'\begin{enumerate}')
        elif line.startswith('### '):body.append(r'\subsection{'+tex_escape(re.sub(r'^\d+\.\d+\s*','',line[4:]))+'}')
        elif line=='\\[':
            block=[line];i+=1
            while i<len(lines):
                block.append(lines[i])
                if lines[i]=='\\]':break
                i+=1
            body.extend(block)
        elif line.startswith('|'):
            rows=[]
            while i<len(lines) and lines[i].startswith('|'):
                cells=[c.strip() for c in lines[i].strip('|').split('|')]
                if not all(re.fullmatch(r'[-: ]+',c) for c in cells):rows.append(cells)
                i+=1
            n=len(rows[0]);body.append(r'\begin{center}\small\resizebox{\linewidth}{!}{\begin{tabular}{'+'l'+'r'*(n-1)+'}'+r'\toprule')
            for j,row in enumerate(rows):body.append(' & '.join(inline(c) for c in row)+r' \\'+(r'\midrule' if j==0 else ''))
            body.append(r'\bottomrule\end{tabular}}\end{center}');continue
        elif line.startswith('!['):
            match=re.search(r'\(([^)]+)\)',line);kind=Path(match.group(1)).stem
            j=i+1
            while j<len(lines) and not lines[j].strip():j+=1
            caption=lines[j].strip('*') if j<len(lines) else kind
            caption=re.sub(r'^Figure \d+\.\s*','',caption)
            body.append(r'\begin{figure}[ht]\centering'+picture(kind,evidence)+r'\caption{'+inline(caption)+r'}\end{figure}')
            i=j
        elif references and re.match(r'^\d+\. ',line):body.append(r'\item '+inline(re.sub(r'^\d+\. ','',line)))
        elif line:body.append(inline(line)+'\n')
        else:body.append('')
        i+=1
    if references:body.append(r'\end{enumerate}')
    title=lines[0][2:]
    header=r'''\documentclass[11pt,a4paper]{article}
\usepackage[margin=23mm]{geometry}
\usepackage[T1]{fontenc}
\usepackage{lmodern,amsmath,booktabs,graphicx,xcolor}
\usepackage[hidelinks]{hyperref}
\setlength{\parindent}{0pt}
\setlength{\parskip}{5pt}
\setlength{\emergencystretch}{3em}
\hypersetup{pdftitle={SELight audited research prototype}}
'''
    return header+r'\title{'+tex_escape(title)+r'}\author{Research prototype draft; author details to be supplied}\date{12 October 2026}\begin{document}\maketitle'+'\n'+'\n'.join(body)+'\n'+r'\end{document}'


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',default='research');a=p.parse_args();out=Path(a.out)
    e=read_json(out/'evidence.json');agg=e['fresh']['aggregate'];methods=list(NAMES)
    tables={}
    tables['FRESH_TABLE']=table(['Method','AP50:95 %','AP50 %','Micro F1','Mean-image F1','Mean ms','P95 ms'],[[NAMES[m],f'{agg[m]["map_50_95"]*100:.3f}',f'{agg[m]["map_50"]*100:.3f}',f'{agg[m]["f1"]:.4f}',f'{agg[m]["mean_image_f1"]:.4f}',f'{agg[m]["mean_ms"]:.2f}',f'{agg[m]["p95_ms"]:.2f}'] for m in methods])
    tables['CI_TABLE']=table(['Candidate','AP delta (pp)','AP 95% interval (pp)','F1 delta'],[[NAMES[m],f'{e["paired"][m]["ap_difference"]*100:+.3f}','['+', '.join(f'{v*100:+.3f}' for v in e['paired'][m]['ap_percentile_95_interval'])+']',f'{e["paired"][m]["mean_image_f1_difference"]:+.4f}'] for m in methods[1:]])
    tables['F1_STRATA_TABLE']=table(['Method','Positive-image F1','Negative-image F1'],[[NAMES[m],f'{e["stratified_fresh_f1"][m]["target_positive_mean_image_f1"]:.4f}',f'{e["stratified_fresh_f1"][m]["target_negative_mean_image_f1"]:.4f}'] for m in methods])
    tables['LEGACY_TABLE']=table(['Device','Method','AP50:95 %','AP50 %','Micro F1','Mean ms','P95 ms'],[[d,NAMES.get(m,m),f'{v["map_50_95"]*100:.3f}',f'{v["map_50"]*100:.3f}',f'{v["f1"]:.4f}',f'{v["mean_ms"]:.2f}',f'{v["p95_ms"]:.2f}'] for d in ['gpu','cpu'] for m,v in e['legacy'][d].items() if 'map_50_95' in v])
    tables['ABLATION_TABLE']=table(['Variant','AP50:95 %','Mean-image F1','Route counts'],[[m,f'{v["map_50_95"]*100:.3f}',f'{v["mean_image_f1"]:.4f}',str(v['route_counts'])] for m,v in e['validation_ablations']['results'].items()])
    path=out/'PAPER.md';paper=path.read_text(encoding='utf-8')
    for name,value in tables.items():
        block=f'<!-- {name}:START -->\n{value}\n<!-- {name}:END -->'
        if '{{'+name+'}}' in paper:paper=paper.replace('{{'+name+'}}',block)
        else:paper=re.sub(r'<!-- '+name+r':START -->.*?<!-- '+name+r':END -->',lambda _:block,paper,flags=re.S)
    if '{{' in paper:raise ValueError('Unfilled manuscript placeholder')
    # Improve prose spacing inherited from compact parameter notation.
    paper=re.sub(r'\b(train|validation|test|utility|lambda|depth|batch|limit|size|version|sigma|quality|percentile|commit|clip|totaling)(?=\d)',r'\1 ',paper)
    paper=re.sub(r'\b(on|from|with|using|of|for|the|all|same|on|by|retained|producing a|reviewed)(?=\d)',r'\1 ',paper)
    path.write_text(paper,encoding='utf-8');(out/'paper.tex').write_text(make_tex(paper,e),encoding='utf-8')
    gallery=read_json(out/'gallery.json')
    tr=''.join('<tr>'+''.join('<td>'+html.escape(c.strip())+'</td>' for c in row.strip('|').split('|'))+'</tr>' for row in tables['FRESH_TABLE'].splitlines() if not row.startswith('| ---'))
    cases=''.join('<article class="case"><h3>'+html.escape(g['id'])+'</h3><p>'+html.escape(g['reason'])+'</p><a href="'+g['file']+'"><img src="'+g['file']+'" alt="Matched ground truth and five-method detection comparison for '+html.escape(g['id'])+'"></a></article>' for g in gallery)
    pair_rows=''.join('<tr>'+''.join('<td>'+html.escape(c.strip())+'</td>' for c in row.strip('|').split('|'))+'</tr>' for row in tables['CI_TABLE'].splitlines() if not row.startswith('| ---'))
    code=(out/'CODE_MAP.md').read_text(encoding='utf-8');tree=re.search(r'```text\n(.*?)```',code,re.S).group(1)
    page='''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>SELight | Research evidence</title><style>
    :root{--ink:#142c3b;--muted:#566874;--paper:#f4f6f5;--teal:#236d66}*{box-sizing:border-box}body{margin:0;background:var(--paper);color:var(--ink);font:16px/1.65 system-ui,sans-serif}main{max-width:1240px;margin:auto;padding:40px 24px}header{border-bottom:1px solid #cbd6d6;padding-bottom:32px}small,.eyebrow{font-weight:700;letter-spacing:.11em;color:var(--teal);text-transform:uppercase}h1{font-size:clamp(36px,5vw,64px);letter-spacing:-.04em;line-height:1.1;max-width:950px;margin:16px 0}h2{font-size:29px;margin:0 0 16px}h3{margin:0}p{max-width:940px}a{color:var(--teal)}nav{display:flex;gap:22px;flex-wrap:wrap;margin-top:24px}section{margin:40px 0;padding:30px;background:white;border:1px solid #dce3e2;border-radius:12px}.cards{display:grid;grid-template-columns:repeat(3,1fr);gap:16px;margin-top:28px}.card{padding:20px;background:#e7efed;border-radius:8px}.card strong{display:block;font-size:32px;line-height:1.25}.note{padding:16px 20px;background:#fff4df;border-left:4px solid #d79835}.table-wrap{overflow:auto}table{width:100%;border-collapse:collapse;white-space:nowrap;font-size:14px}td{border-bottom:1px solid #e3e8e7;padding:11px 12px;text-align:right}td:first-child{text-align:left}tr:first-child{font-weight:700;background:#edf3f1}img{display:block;max-width:100%;height:auto}.grid{display:grid;grid-template-columns:1fr 1fr;gap:25px}.case{padding:24px 0;border-top:1px solid #e3e8e7}.case p{color:var(--muted);margin-top:4px}.case img{border:1px solid #e3e8e7}pre{overflow:auto;font:13px/1.6 Consolas,monospace;background:#142c3b;color:#e2ece9;padding:24px;border-radius:8px}footer{color:var(--muted);font-size:13px;padding:16px}button{background:var(--teal);border:0;border-radius:5px;color:white;padding:10px 15px;cursor:pointer}@media(max-width:740px){main{padding:24px 12px}.cards,.grid{grid-template-columns:1fr}section{padding:18px}nav{gap:14px}}@media print{body{background:white}main{max-width:none;padding:0}section{break-inside:avoid;border:0}nav,button{display:none}}
    </style></head><body><main><header><div class="eyebrow">Research prototype / 12 October 2026</div><h1>What selective enhancement actually achieves.</h1><p>SELight investigates one inexpensive input transform before a frozen YOLOv8n detector. This showcase links every measured claim to matching images, raw predictions, timing scope and source provenance.</p><nav><a href="../output/pdf/SELight_research_paper.pdf">Paper PDF</a><a href="PAPER.md">Manuscript</a><a href="paper.tex">LaTeX source</a><a href="RESEARCH_CONTEXT.md">Full context</a><a href="CODE_MAP.md">Code map</a><a href="RUNBOOK.md">Run commands</a><a href="comparison.csv">CSV</a><a href="evidence.json">Evidence JSON</a></nav><div class="cards"><div class="card"><small>Paired validation</small><strong>200 images</strong>5 methods / 3 timing repeats</div><div class="card"><small>Original and SELight AP</small><strong>43.02%</strong>Identical prediction dictionaries</div><div class="card"><small>Software verification</small><strong>31 tests pass</strong>CUDA / RTX 4050 laptop GPU</div></div></header>
    <section><h2>The current result</h2><div class="note">The learned policy always chooses original input. It adds measured processing cost without changing detections. The current evidence supports a reproducible prototype and negative result; it does not demonstrate adaptive superiority.</div><img src="figures/pipeline.png" alt="SELight inference pipeline"></section>
    <section id="comparison"><div class="eyebrow">Fresh matched inference</div><h2>Quality and cost on the same inputs</h2><p>Five-class ExDark validation subset, frozen checkpoint, FP32, imgsz 640. All methods see identical image IDs and annotations. Accuracy uses repeat 0; repeats are cost measurements, not independent accuracy samples.</p><img src="figures/fresh_comparison.png" alt="Five-method AP and processing time"><div class="table-wrap"><table>'''+tr+'''</table></div><p>Processing timing includes selection/enhancement, transfers, output quantization and synchronized detection. Decode, startup, acquisition, drawing and writing are excluded. Zero-DCE++ uses a geometry-preserving padding adaptation.</p></section>
    <section><h2>Paired uncertainty</h2><img src="figures/paired_intervals.png" alt="Paired AP confidence intervals"><div class="table-wrap"><table>'''+pair_rows+'''</table></div><p>1,000 paired image resamples, dataset-level COCO AP recomputation; independent images assumed because capture-session labels are unavailable. Intervals crossing zero establish neither superiority nor equivalence. The identical selector has an analytically zero interval.</p></section>
    <section><h2>Historical evidence has a separate scope</h2><div class="grid"><div><img src="figures/legacy_per_class.png" alt="Recomputed historical per-class AP"><p>2,560 stored test images. Existing predictions, metrics recomputed now; no new test inference.</p></div><div><img src="figures/stress_pilot.png" alt="Historical synthetic stress test"><p>30 validation images under synthetic resize/noise/JPEG conditions. This does not establish performance on inexpensive camera sensors.</p></div></div><p>Legacy timings include decode. A roughly 69.6-minute CPU decode interruption remains in the raw selector mean. Full test, fresh validation and stress results must not be mixed into one ranking.</p></section>
    <section id="gallery"><div class="eyebrow">Successes and failures</div><h2>Inspect the actual detections</h2><p>Yellow dashed boxes are ground truth; cyan solid boxes are predictions at confidence &ge; .25. Cases follow declared diagnostic rules rather than a success-only selection. Click a strip to inspect it at full resolution.</p>'''+cases+'''</section>
    <section><h2>A readable implementation tree</h2><pre>'''+html.escape(tree)+'''</pre><p>Production files remain in their working locations. The new research package groups context, plots and reproducible comparisons without changing the frozen algorithm.</p></section>
    <section><h2>Read the limits before presenting the result</h2><p>The current policy has no demonstrated quality/cost advantage. Validation was already used for policy selection. Near-duplicate/annotation-origin uncertainty remains. SCI and Retinexformer are cited related work, not locally benchmarked. Genuine labelled phone sessions, an untouched session test, Android execution and calibrated lux are not available.</p><p>The gallery uses ExDark source photographs for local academic inspection; their underlying image rights are separate from the project code. Dataset and method attribution: <a href="https://github.com/cs-chan/Exclusively-Dark-Image-Dataset">ExDark</a>, <a href="https://docs.ultralytics.com/models/yolov8/">YOLOv8</a>, <a href="https://github.com/Li-Chongyi/Zero-DCE_extension">official Zero-DCE++</a>. See the manuscript for full primary references.</p></section>
    <footer>Generated from local evidence. No external JavaScript, tracking, publication or hosting. Raw inputs are hash-bound in evidence.json.</footer></main></body></html>'''
    (out/'showcase.html').write_text(page,encoding='utf-8')
    root=Path.cwd()
    paths=sorted(root.glob('*.py'))+sorted((root/'selight').glob('*.py'))+sorted((root/'scripts').glob('*.py'))+sorted((root/'tests').glob('*.py'))
    inventory=[]
    for path in paths:
        text=path.read_text(encoding='utf-8-sig');t=ast.parse(text)
        definitions=[dict(name=n.name,line=n.lineno,kind=type(n).__name__,doc=ast.get_docstring(n)) for n in t.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))]
        imports=sorted({n.module or '' for n in ast.walk(t) if isinstance(n,ast.ImportFrom)}|{z.name for n in ast.walk(t) if isinstance(n,ast.Import) for z in n.names})
        inventory.append(dict(path=str(path.relative_to(root)).replace('\\','/'),lines=len(text.splitlines()),sha256=sha(path),definitions=definitions,imports=imports,module_doc=ast.get_docstring(t)))
    write_json(out/'code_inventory.json',dict(files=inventory,total_python_lines=sum(x['lines'] for x in inventory),scope='Owned Python source and tests, excluding environment and third-party code'))
    write_json(out/'artifact_hashes.json',{str(x.relative_to(out)).replace('\\','/'):sha(x) for x in out.rglob('*') if x.is_file() and x.name!='artifact_hashes.json'})
    print('Manuscript tables, standalone TeX and showcase rendered.')


if __name__=='__main__':main()
