"""Export the authored, verified Markdown paper to a shareable PDF.

Uses reportlab and pypdf from the bundled document runtime. This is a Markdown
export, not a claimed compilation of paper.tex. Figures are embedded from PNGs.
"""
import argparse
import hashlib
import html
import json
import re
from pathlib import Path
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, KeepTogether, PageBreak
from pypdf import PdfReader


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rich(text):
    text=text.replace('—','-').replace('–','-').replace('\u2011','-')
    parts=[];start=0
    for m in re.finditer(r'\[([^\]]+)\]\(([^)]+)\)|`([^`]+)`|\*\*([^*]+)\*\*|\*([^*]+)\*',text):
        parts.append(html.escape(text[start:m.start()]))
        if m.group(1):parts.append('<a color="#236d66" href="'+html.escape(m.group(2),quote=True)+'">'+html.escape(m.group(1))+'</a>')
        elif m.group(3):parts.append('<font name="Courier" size="8.3">'+html.escape(m.group(3))+'</font>')
        elif m.group(4):parts.append('<b>'+html.escape(m.group(4))+'</b>')
        else:parts.append('<i>'+html.escape(m.group(5))+'</i>')
        start=m.end()
    parts.append(html.escape(text[start:]));return ''.join(parts)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',default='research/PAPER.md')
    p.add_argument('--out',default='output/pdf/SELight_research_paper.pdf')
    p.add_argument('--font-dir',default='C:/Windows/Fonts')
    a=p.parse_args();source=Path(a.source);output=Path(a.out);output.parent.mkdir(parents=True,exist_ok=True)
    fonts=Path(a.font_dir)
    for name,file in [('ResearchTimes','times.ttf'),('ResearchTimesBold','timesbd.ttf'),('ResearchTimesItalic','timesi.ttf'),('ResearchTimesBoldItalic','timesbi.ttf')]:
        pdfmetrics.registerFont(TTFont(name,str(fonts/file)))
    pdfmetrics.registerFontFamily('ResearchTimes',normal='ResearchTimes',bold='ResearchTimesBold',italic='ResearchTimesItalic',boldItalic='ResearchTimesBoldItalic')
    styles=getSampleStyleSheet()
    styles.add(ParagraphStyle('ResearchBody',fontName='ResearchTimes',fontSize=10.5,leading=14.2,spaceAfter=7,alignment=TA_JUSTIFY,splitLongWords=True))
    styles.add(ParagraphStyle('ResearchTitle',fontName='ResearchTimesBold',fontSize=21,leading=25,spaceAfter=16,textColor=colors.HexColor('#142c3b')))
    styles.add(ParagraphStyle('ResearchHeading',fontName='ResearchTimesBold',fontSize=14,leading=18,spaceBefore=14,spaceAfter=8,keepWithNext=True,textColor=colors.HexColor('#142c3b')))
    styles.add(ParagraphStyle('ResearchSubheading',fontName='ResearchTimesBold',fontSize=11.5,leading=15,spaceBefore=10,spaceAfter=5,keepWithNext=True))
    styles.add(ParagraphStyle('ResearchCaption',fontName='ResearchTimesItalic',fontSize=9,leading=11.5,spaceAfter=10,textColor=colors.HexColor('#46535a')))
    styles.add(ParagraphStyle('ResearchCell',fontName='ResearchTimes',fontSize=8.2,leading=10.3,splitLongWords=True))
    styles.add(ParagraphStyle('ResearchFormula',fontName='ResearchTimes',fontSize=10.5,leading=18,alignment=TA_CENTER,spaceBefore=8,spaceAfter=12))
    doc=SimpleDocTemplate(str(output),pagesize=A4,rightMargin=48,leftMargin=48,topMargin=47,bottomMargin=47,
        title='SELight: audited enhancement-selection research study',author='Research prototype - author details to be supplied')
    lines=source.read_text(encoding='utf-8').splitlines();story=[];i=0;formula_number=0
    while i<len(lines):
        line=lines[i].strip()
        if not line or line.startswith('<!--'):i+=1;continue
        if line.startswith('# '):story.append(Paragraph(rich(line[2:]),styles['ResearchTitle']))
        elif line.startswith('## '):
            if line[3:]=='References':story.append(PageBreak())
            story.append(Paragraph(rich(line[3:]),styles['ResearchHeading']))
        elif line.startswith('### '):story.append(Paragraph(rich(line[4:]),styles['ResearchSubheading']))
        elif line=='\\[':
            while i<len(lines) and lines[i].strip()!='\\]':i+=1
            formula_number+=1
            if formula_number==1:
                text='c<sub>r</sub> = max(0, median<sub>i in T</sub>(t<sub>i,r</sub>)/median<sub>i in T</sub>(t<sub>i,o</sub>) - 1), &nbsp; c<sub>o</sub> = 0.'
            else:
                text='y<sub>i</sub> = argmax [q(i,r) - λc<sub>r</sub>] for r in E<sub>i</sub>,<br/>E<sub>i</sub> = {o} union {r : q(i,r) ≥ q(i,o) + 0.02}.'
            story.append(Paragraph(text,styles['ResearchFormula']))
        elif line.startswith('|'):
            rows=[]
            while i<len(lines) and lines[i].strip().startswith('|'):
                cells=[c.strip() for c in lines[i].strip().strip('|').split('|')]
                if not all(re.fullmatch(r'[-: ]+',c) for c in cells):rows.append([Paragraph(rich(c),styles['ResearchCell']) for c in cells])
                i+=1
            n=len(rows[0]);first=95 if n>4 else 160;other=(doc.width-first)/(n-1)
            t=Table(rows,colWidths=[first]+[other]*(n-1),repeatRows=1,hAlign='LEFT')
            t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e7efed')),('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),5),('RIGHTPADDING',(0,0),(-1,-1),5),('TOPPADDING',(0,0),(-1,-1),5),('BOTTOMPADDING',(0,0),(-1,-1),5),('LINEABOVE',(0,0),(-1,0),.6,colors.HexColor('#566874')),('LINEBELOW',(0,0),(-1,0),.4,colors.HexColor('#566874')),('LINEBELOW',(0,-1),(-1,-1),.5,colors.HexColor('#566874')),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#f5f7f6')])]))
            story.extend([t,Spacer(1,7)]);continue
        elif line.startswith('!['):
            path=source.parent/re.search(r'\(([^)]+)\)',line).group(1)
            image=Image(str(path));ratio=image.imageHeight/image.imageWidth;image.drawWidth=doc.width;image.drawHeight=doc.width*ratio
            if image.drawHeight>260:image.drawHeight=260;image.drawWidth=260/ratio
            j=i+1
            while j<len(lines) and not lines[j].strip():j+=1
            caption=lines[j].strip().strip('*') if j<len(lines) else path.stem
            story.append(KeepTogether([Spacer(1,8),image,Spacer(1,5),Paragraph(rich(caption),styles['ResearchCaption'])]));i=j
        elif line.startswith('*Table '):story.append(Paragraph(rich(line.strip('*')),styles['ResearchCaption']))
        else:story.append(Paragraph(rich(line),styles['ResearchBody']))
        i+=1

    def page(canvas,doc):
        canvas.saveState();canvas.setStrokeColor(colors.HexColor('#c8d2d3'));canvas.line(48,31,A4[0]-48,31)
        canvas.setFont('ResearchTimes',8);canvas.setFillColor(colors.HexColor('#566874'))
        canvas.drawString(48,20,'SELight | Audited prototype research study | 12 October 2026')
        canvas.drawRightString(A4[0]-48,20,str(doc.page));canvas.restoreState()
    doc.build(story,onFirstPage=page,onLaterPages=page)
    reader=PdfReader(str(output));text='\n'.join(p.extract_text() or '' for p in reader.pages)
    expected=['43.019','42.328','30.23','References','69.6','1,000']
    if 'Frozen external dataset transfer' in source.read_text(encoding='utf-8'):
        expected += ['39.395','85.362','423','78','LLVIP','Penn-Fudan']
    missing=[v for v in expected if v not in text]
    if missing:raise ValueError('Missing PDF text: '+str(missing))
    validation=dict(pages=len(reader.pages),missing_expected_text=missing,source_sha256=digest(source),pdf_sha256=digest(output),
                    export='ReportLab from verified Markdown, not native LaTeX compilation',requires_visual_review=True)
    (source.parent/'pdf_validation.json').write_text(json.dumps(validation,indent=2),encoding='utf-8')
    print(json.dumps(validation))


if __name__=='__main__':main()
