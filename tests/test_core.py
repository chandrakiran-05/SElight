import json
import numpy as np
import pytest
from selight.core import Enhancer, features, tree_route, select
from selight.metrics import counts, quality, summary
from selight.data import parse_boxes, audit
from selight.experiment import train, route_label, signature

@pytest.fixture
def cfg():
    return json.load(open('configs/default.json'))

def test_gamma_brightens_without_geometry_change(cfg):
    a=np.full((40,60,3),64,np.uint8); b=Enhancer(cfg).apply(a,'gamma')
    assert b.shape==a.shape and b.dtype==np.uint8 and b.mean()>a.mean()
    assert np.all(a==64)

@pytest.mark.parametrize('route',['original','gamma','clahe'])
def test_routes_preserve_bounds(cfg,route):
    a=np.random.default_rng(5).integers(0,256,(80,120,3),dtype=np.uint8)
    b=Enhancer(cfg).apply(a,route)
    assert b.shape==a.shape and b.dtype==np.uint8

def test_extreme_images_have_finite_features(cfg):
    for v in [0,255]:
        f=features(np.full((20,30,3),v,np.uint8))
        assert len(f)==8 and np.isfinite(f).all()

def test_class_and_duplicate_matching():
    gt=[[0,0,0,10,10]]
    pred=[[0,.9,0,0,10,10],[0,.8,0,0,10,10],[1,.95,0,0,10,10]]
    assert counts(gt,pred)==(1,2,0)
    assert quality(gt,[])==0 and quality([],[])==1

def test_iou_threshold_and_confidence():
    assert counts([[0,0,0,10,10]],[[0,.1,0,0,10,10]])==(0,0,1)
    assert counts([[0,0,0,10,10]],[[0,.9,8,8,18,18]])==(0,1,1)

def test_annotation_origin_and_multiple_classes():
    text='% bbGt version=3\nPeople 1 1 10 20 0 0\nCar 3 4 2 3\nDog 2 2 2 2'
    b=parse_boxes(text,30,40,1)
    assert b==[[0,0.,0.,10.,20.],[2,2.,3.,4.,6.]]
    with pytest.raises(ValueError): parse_boxes('People 1 1 -3 5',10,10,0)

def test_cross_split_duplicates_rejected():
    rows=[{'id':s,'split':s,'sha256':'same','path':'x','boxes':[],'width':20,'height':20} for s in ['train','val','test']]
    with pytest.raises(ValueError): audit({'root':'.','records':rows},False)

def test_quality_margin_and_cost():
    q={'original':.5,'gamma':.51,'clahe':.6}; costs={'original':0,'gamma':0,'clahe':10}
    assert route_label(q,costs,0)=='clahe'
    assert route_label(q,costs,.05)=='original'

def test_train_export_and_no_test_leakage(tmp_path,cfg):
    rows=[]
    for split in ['train','val']:
        for i in range(30):
            dark=i%2==0
            rows.append({'id':f'{split}{i}','sha256':f'{split}{i}','split':split,
                'features':[.1 if dark else .8]+[0]*7,'quality':{'original':.4 if dark else 1.,'gamma':.9 if dark else .6,'clahe':.3},
                'ms':{'original':10.,'gamma':11.,'clahe':13.}})
    cache={'signature':'test','manifest_hash':'test','records':rows,'environment':{'config':{'device':'cpu'}}}
    p=tmp_path/'cache.json'; p.write_text(json.dumps(cache)); out=tmp_path/'policy.json'
    train(p,out); policy=json.loads(out.read_text())
    assert tree_route(policy['tree'],[.1]+[0]*7)=='gamma'
    assert tree_route(policy['tree'],[.8]+[0]*7)=='original'
    assert select(np.zeros((32,32,3),np.uint8),'selector',cfg,policy)=='gamma'
    rows[0]['split']='test'; p.write_text(json.dumps(cache))
    with pytest.raises(ValueError): train(p,out)

def test_coco_perfect_and_empty():
    rows=[{'id':'a','width':100,'height':100,'boxes':[[0,0,0,20,20]]}]
    perfect=summary(rows,{'a':[[0,.9,0,0,20,20]]})
    empty=summary(rows,{'a':[]})
    assert perfect['map_50_95']>.99 and perfect['f1']==1
    assert empty['map_50_95']==0 and empty['fn']==1

def test_signature_checks_model_not_device(cfg):
    b={**cfg,'device':'0'}
    assert signature(cfg,'a')==signature(b,'a')
    assert signature(cfg,'a')!=signature(cfg,'b')
