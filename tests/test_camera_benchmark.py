import cv2
import numpy as np
from camera_benchmark import simulate_input, restore_boxes, CONDITIONS, run
from selight.core import read_json
from selight.data import prepare
from selight.experiment import build_cache, train


def test_stress_deterministic_no_mutation():
    image=np.full((100,201,3),50,np.uint8); before=image.copy()
    c={'max_side':80,'noise_sigma':5.,'jpeg_quality':50}
    a,ratio=simulate_input(image,c,'a'); b,_=simulate_input(image,c,'a')
    assert np.array_equal(a,b) and np.array_equal(image,before)
    assert a.shape[:2]==(40,80)
    assert ratio==(80/201,40/100)


def test_restore_geometry_and_native_identity():
    image=np.full((100,200,3),50,np.uint8)
    a,r=simulate_input(image,CONDITIONS['native'],'x')
    assert np.array_equal(a,image) and r==(1.,1.)
    assert restore_boxes([[0,.9,10.,5.,30.,25.]],(.5,.25))==[[0,.9,20.,20.,60.,100.]]


def test_no_upsample_small_image():
    a,r=simulate_input(np.zeros((10,20,3),np.uint8),CONDITIONS['edge320'],'x')
    assert a.shape==(10,20,3) and r==(1.,1.)


def test_camera_fixture(tmp_path,monkeypatch):
    import selight.detector
    class FakeDetector:
        def __init__(self,cfg): self.weight_hash='fixture'; self.cfg=cfg
        def sync(self): pass
        def predict(self,image): return [[0,.9,0.,0.,10.,10.]]
    monkeypatch.setattr(selight.detector,'Detector',FakeDetector)
    images=tmp_path/'images'; anns=tmp_path/'ann'; images.mkdir(); anns.mkdir()
    lines=[]
    for split in (1,2,3):
        for i in range(3):
            name=f'{split}_{i}.png'; cv2.imwrite(str(images/name),np.full((30,40,3),20+split+i*35,np.uint8))
            (anns/(name+'.txt')).write_text('% bbGt version=3\nPeople 1 1 10 10 0 0 0\n')
            lines.append(f'{name} 11 1 2 {split}')
    sf=tmp_path/'split.txt'; sf.write_text('\n'.join(lines))
    manifest=tmp_path/'manifest.json'; prepare(images,anns,sf,manifest,1)
    cfg=read_json('configs/default.json'); cache=tmp_path/'cache.json'; policy=tmp_path/'policy.json'
    build_cache(manifest,cfg,cache); train(cache,policy)
    run(manifest,'configs/default.json',policy,tmp_path/'stress',split='test',repeats=1,condition_names=['native','edge320'])
    result=read_json(tmp_path/'stress/summary.json')
    assert len(result['results'])==14
    assert all(sum(v['route_counts'].values())==3 for v in result['results'].values())
    assert all(v['map_50']>.99 for v in result['results'].values())
