"""Detection operating-point scores and standard COCO AP, not proxy AP."""
import numpy as np
from .core import NAMES

def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2-x1)*max(0, y2-y1)
    area = max(0,a[2]-a[0])*max(0,a[3]-a[1])+max(0,b[2]-b[0])*max(0,b[3]-b[1])-inter
    return inter/area if area else 0.

def counts(gt, pred, conf=.25):
    # gt: [class,x1,y1,x2,y2]; pred: [class,score,x1,y1,x2,y2].
    pred = sorted((p for p in pred if p[1] >= conf), key=lambda p: -p[1])
    used = set(); tp = 0
    for p in pred:
        options = [(iou(p[2:], g[1:]), j) for j, g in enumerate(gt) if j not in used and int(g[0]) == int(p[0])]
        if options:
            overlap, j = max(options)
            if overlap >= .5: used.add(j); tp += 1
    return tp, len(pred)-tp, len(gt)-tp

def f1_counts(tp, fp, fn):
    return 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 1.

def quality(gt, pred, conf=.25):
    # Empty GT and empty prediction has utility 1; AP is evaluated separately.
    return f1_counts(*counts(gt, pred, conf))

def summary(records, predictions, conf=.25):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    dataset = {"info": {}, "licenses": [], "images": [], "annotations": [],
               "categories": [{"id":i,"name":n} for i,n in enumerate(NAMES)]}
    detections=[]; aid=1; totals=np.zeros(3, dtype=int)
    for i, r in enumerate(records):
        dataset["images"].append({"id":i,"width":r["width"],"height":r["height"]})
        for c,x1,y1,x2,y2 in r["boxes"]:
            dataset["annotations"].append({"id":aid,"image_id":i,"category_id":int(c),
              "bbox":[x1,y1,x2-x1,y2-y1],"area":(x2-x1)*(y2-y1),"iscrowd":0}); aid+=1
        pred=predictions[r["id"]]
        totals += counts(r["boxes"], pred, conf)
        for c,s,x1,y1,x2,y2 in pred:
            detections.append({"image_id":i,"category_id":int(c),"score":s,"bbox":[x1,y1,x2-x1,y2-y1]})
    coco=COCO(); coco.dataset=dataset; coco.createIndex()
    if detections:
        dt=coco.loadRes(detections)
    else:
        dt=COCO(); dt.dataset={**dataset,"annotations":[]}; dt.createIndex()
    ev=COCOeval(coco,dt,"bbox"); ev.params.imgIds=list(range(len(records)))
    ev.evaluate(); ev.accumulate(); ev.summarize()
    tp,fp,fn=map(int,totals)
    per_class={}
    for k,name in enumerate(NAMES):
        values=ev.eval['precision'][:,:,k,0,-1]
        valid=values[values>=0]
        per_class[name]=float(valid.mean()) if valid.size else -1.
    return {"map_50_95":float(ev.stats[0]),"map_50":float(ev.stats[1]),
            "per_class_ap":per_class,
            "mean_image_f1":float(np.mean([quality(r['boxes'],predictions[r['id']],conf) for r in records])),
            "ap_small":float(ev.stats[3]),"precision":tp/(tp+fp) if tp+fp else 0.,
            "recall":tp/(tp+fn) if tp+fn else 0.,"f1":f1_counts(tp,fp,fn),
            "tp":tp,"fp":fp,"fn":fn,"images":len(records),
            "note":"AP -1 means no eligible annotations; AP is COCO style on the declared five-class subset."}
