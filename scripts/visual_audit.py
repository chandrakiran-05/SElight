"""Perceptual duplicate candidates and validation annotation contact sheets."""
from pathlib import Path
import argparse
import cv2
import numpy as np
from selight.core import read_json,write_json,load_image,sha,NAMES
from selight.data import index_files


def tile(record,obj,label,boxes=False):
    im=load_image(Path(obj['root'])/record['path'])
    if boxes:
        for c,x1,y1,x2,y2 in record['boxes']:
            cv2.rectangle(im,(round(x1),round(y1)),(round(x2),round(y2)),(0,255,0),2)
            cv2.putText(im,NAMES[c],(round(x1),max(15,round(y1))),0,.5,(0,255,0),1)
    h,w=im.shape[:2]; scale=min(360/w,250/h)
    thumb=cv2.resize(im,(round(w*scale),round(h*scale)),interpolation=cv2.INTER_AREA)
    canvas=np.full((290,360,3),30,np.uint8); canvas[:thumb.shape[0],:thumb.shape[1]]=thumb
    cv2.putText(canvas,label[:48],(5,273),0,.4,(255,255,255),1)
    return canvas

def save(path,image):
    ok,data=cv2.imencode('.jpg',image,[cv2.IMWRITE_JPEG_QUALITY,90])
    if not ok: raise ValueError('Encoding failure')
    data.tofile(str(path))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',required=True); p.add_argument('--out',required=True); args=p.parse_args()
    OUT=Path(args.out); OUT.mkdir(parents=True,exist_ok=False); obj=read_json(args.manifest); rows=obj['records']
    codes=[]; mismatch=[]
    # Perceptual DCT hashes flag candidates; they do not justify automatic removal.
    for i,r in enumerate(rows):
        im=load_image(Path(obj['root'])/r['path']); h,w=im.shape[:2]
        if (w,h)!=(r['width'],r['height']): mismatch.append(r['id'])
        gray=cv2.cvtColor(cv2.resize(im,(32,32),interpolation=cv2.INTER_AREA),cv2.COLOR_BGR2GRAY).astype(np.float32)
        dct=cv2.dct(gray)[:8,:8].reshape(-1); bits=dct>np.median(dct[1:]); bits[0]=False
        codes.append(np.packbits(bits))
    codes=np.array(codes,dtype=np.uint8); lut=np.array([int(i).bit_count() for i in range(256)],dtype=np.uint8)
    pairs=[]
    for start in range(0,len(rows),64):
        distance=lut[np.bitwise_xor(codes[start:start+64,None,:],codes[None,:,:])].sum(axis=2)
        for i,j in zip(*np.where(distance<=4)):
            i+=start
            if i<j and rows[i]['split']!=rows[j]['split']:
                pairs.append({'first':rows[i]['id'],'second':rows[j]['id'],'first_split':rows[i]['split'],'second_split':rows[j]['split'],'hamming':int(distance[i-start,j])})
    pairs.sort(key=lambda p:(p['hamming'],p['first'],p['second']))
    byid={r['id']:r for r in rows}
    write_json(OUT/'near_duplicate_candidates.json',{'manifest_hash':sha(args.manifest),'definition':'32x32 grayscale INTER_AREA; 8x8 DCT median hash excluding DC; Hamming <=4; cross-split only; manual review required','dimension_mismatches':mismatch,'candidates':pairs})
    for chunk in range(0,min(len(pairs),48),8):
        tiles=[]
        for p in pairs[chunk:chunk+8]:
            tiles.append(np.hstack([tile(byid[p['first']],obj,f"{p['first_split']} {p['first']} d={p['hamming']}"),tile(byid[p['second']],obj,f"{p['second_split']} {p['second']}")]))
        save(OUT/f'near_duplicates_{chunk//8:02d}.jpg',np.vstack(tiles))
    # Stratify overlays over five classes, lighting, small objects, and negatives, validation only.
    val=[r for r in rows if r['split']=='val']; selected=[]; used=set()
    groups=[[r for r in val if any(b[0]==c for b in r['boxes'])] for c in range(5)]
    groups += [[r for r in val if r['lighting']==light] for light in range(1,11)]
    groups += [[r for r in val if not r['boxes']],[r for r in val if any((b[3]-b[1])*(b[4]-b[2])<1024 for b in r['boxes'])]]
    for group in groups:
        for r in sorted(group,key=lambda r:r['sha256']):
            if r['id'] not in used: selected.append(r); used.add(r['id']); break
    for r in sorted(val,key=lambda r:r['sha256']):
        if len(selected)>=30: break
        if r['id'] not in used: selected.append(r); used.add(r['id'])
    for start in range(0,len(selected),6):
        tiles=[tile(r,obj,f"{r['id']} light={r['lighting']} boxes={len(r['boxes'])}",True) for r in selected[start:start+6]]
        save(OUT/f'annotations_{start//6:02d}.jpg',np.vstack([np.hstack(tiles[i:i+3]) for i in range(0,len(tiles),3)]))
    write_json(OUT/'annotation_samples.json',{'ids':[r['id'] for r in selected],'coordinate_origin':obj['coordinate_origin'],'note':'Visual overlays can detect gross shifts; a one-pixel origin difference may remain ambiguous.'})
    print('Cross-split perceptual candidates:',len(pairs),'dimension mismatches:',len(mismatch))

if __name__=='__main__': main()
