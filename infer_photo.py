"""Single-photo SELight demo. Outputs detections and annotated image, not timing claims."""
import argparse
from pathlib import Path
import cv2
from selight.core import Enhancer, select, load_image, read_json, write_json, NAMES
from selight.experiment import signature


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--image',required=True); p.add_argument('--policy',default='models/policy.json')
    p.add_argument('--config',default='configs/default.json'); p.add_argument('--out',required=True)
    a=p.parse_args()
    from selight.detector import Detector
    cfg,policy=read_json(a.config),read_json(a.policy); det=Detector(cfg)
    if signature(cfg,det.weight_hash)!=policy['signature']: raise ValueError('Policy/config/checkpoint mismatch')
    image=load_image(a.image); route=select(image,'selector',cfg,policy)
    processed=Enhancer(cfg).apply(image,route); predictions=det.predict(processed)
    out=Path(a.out); out.mkdir(parents=True,exist_ok=False)
    write_json(out/'detections.json',{'route':route,'predictions':predictions,
        'format':['class','score','x1','y1','x2','y2'],'classes':NAMES,
        'note':'Demo only. Without ground truth, detections do not establish accuracy.'})
    canvas=processed.copy()
    for c,s,x1,y1,x2,y2 in predictions:
        if s<cfg['operating_confidence']: continue
        cv2.rectangle(canvas,(int(x1),int(y1)),(int(x2),int(y2)),(0,255,0),2)
        cv2.putText(canvas,f'{NAMES[c]} {s:.2f}',(int(x1),max(18,int(y1))),0,.5,(0,255,0),1)
    ok,data=cv2.imencode('.png',canvas)
    if not ok: raise RuntimeError('Cannot encode output')
    data.tofile(str(out/'annotated.png'))

if __name__=='__main__': main()
