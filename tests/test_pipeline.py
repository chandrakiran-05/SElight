"""Synthetic integration test only. Does not validate a pretrained detector."""
from pathlib import Path
import json
import cv2
import numpy as np
from selight.core import sha, read_json, write_json
from selight.data import prepare, preview
from selight.experiment import build_cache, train, evaluate

def test_full_fixture_pipeline(tmp_path,monkeypatch):
    import selight.detector
    class FakeDetector:
        def __init__(self,cfg): self.weight_hash='fixture'; self.cfg=cfg
        def sync(self): pass
        def predict(self,image):
            return [[0,.9,0.,0.,10.,10.]] if image.mean()>60 else []
    monkeypatch.setattr(selight.detector,'Detector',FakeDetector)
    images=tmp_path/'images'; anns=tmp_path/'ann'; images.mkdir(); anns.mkdir()
    lines=[]
    for split in [1,2,3]:
        for i in range(4):
            name=f'{split}_{i}.png'
            im=np.full((30,40,3),25+split+i*35,np.uint8)
            cv2.imwrite(str(images/name),im)
            (anns/(name+'.txt')).write_text('% bbGt version=3\nPeople 1 1 10 10 0 0 0\n')
            lines.append(f'{name} 11 1 2 {split}')
    splitfile=tmp_path/'split.txt'; splitfile.write_text('\n'.join(lines))
    manifest=tmp_path/'manifest.json'
    prepare(images,anns,splitfile,manifest,1)
    preview(manifest,tmp_path/'preview',3)
    cfg=read_json('configs/default.json'); cache=tmp_path/'cache.json'; policy=tmp_path/'policy.json'
    build_cache(manifest,cfg,cache); train(cache,policy)
    assert not any(r['split']=='test' for r in read_json(cache)['records'])
    evaluate(manifest,cfg,policy,tmp_path/'evaluation',repeats=1)
    result=read_json(tmp_path/'evaluation/summary.json')
    assert len(result['results'])==7
    assert all(r['images']==4 for r in result['results'].values())
    assert (tmp_path/'evaluation/quality_latency.png').exists()
