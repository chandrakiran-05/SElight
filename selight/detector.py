"""Ultralytics adapter; detector import stays optional for offline unit tests."""
from pathlib import Path
from .core import COCO_IDS, sha

class Detector:
    def __init__(self,cfg):
        import torch
        from ultralytics import YOLO
        self.torch=torch; self.cfg=cfg
        torch.set_num_threads(cfg['threads'])
        if cfg['device']!='cpu' and not torch.cuda.is_available():
            raise RuntimeError('CUDA requested but unavailable; fix install or explicitly select cpu')
        if not Path(cfg['weights']).is_file():
            raise FileNotFoundError('Download the official checkpoint first; set weights to a local .pt path')
        self.weight_hash=sha(cfg['weights'])
        self.model=YOLO(cfg['weights'])
        expected={0:'person',1:'bicycle',2:'car',3:'motorcycle',5:'bus'}
        for i,name in expected.items():
            if self.model.names[i]!=name: raise ValueError('Expected COCO class order; use the supplied COCO checkpoint')

    def sync(self):
        if self.cfg['device']!='cpu': self.torch.cuda.synchronize()

    def predict(self,image):
        r=self.model.predict(image,imgsz=self.cfg['imgsz'],device=self.cfg['device'],
             conf=self.cfg['confidence'],iou=self.cfg['nms_iou'],max_det=self.cfg['max_det'],
             classes=list(COCO_IDS),half=False,augment=False,verbose=False)[0]
        b=r.boxes
        if b is None: return []
        xy=b.xyxy.cpu().numpy().tolist(); scores=b.conf.cpu().numpy().tolist(); cs=b.cls.cpu().numpy().tolist()
        return [[COCO_IDS.index(int(c)),float(s),*map(float,box)] for c,s,box in zip(cs,scores,xy)]
