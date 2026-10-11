"""Reference/production equivalence and single-call semantics, not accuracy evidence."""
import copy
import numpy as np
import pytest
from algorithm_reference import choose_training_route, fit_reference_policy, infer_one
from selight.core import ROUTES, Enhancer, read_json, write_json, tree_route
from selight.experiment import route_label, train


def fixture_cache():
    rows=[]
    for split in ('train','val'):
        for i in range(36):
            target=ROUTES[i%3]
            rows.append(dict(id=f'{split}-{i}',sha256=f'{split}-{i}',split=split,
                features=[i%3/3., i/36.,0.,0.,0.,0.,0.,0.],
                quality={r:0.85 if r==target else 0.4 for r in ROUTES},
                ms={'original':10.,'gamma':11.,'clahe':13.}))
    return dict(records=rows,signature='fixture',manifest_hash='fixture',
                environment={'config':{'device':'cpu'}})


def test_label_matches_production():
    rng=np.random.default_rng(15)
    for _ in range(100):
        q=dict(zip(ROUTES,rng.random(3)))
        c=dict(zip(ROUTES,rng.random(3))); c['original']=0.
        for lam in (0.,.01,.05):
            assert choose_training_route(q,c,lam)==route_label(q,c,lam)


def test_margin_cost_and_tie():
    c={'original':0.,'gamma':1.,'clahe':1.}
    assert choose_training_route(dict(original=.5,gamma=.51,clahe=.49),c,0.)=='original'
    assert choose_training_route(dict(original=.5,gamma=.8,clahe=.8),c,0.)=='gamma'
    assert choose_training_route(dict(original=.5,gamma=.6,clahe=.6),c,.2)=='original'


def test_reference_matches_exported_policy(tmp_path):
    cache=fixture_cache(); reference=fit_reference_policy(cache)
    src=tmp_path/'cache.json'; out=tmp_path/'policy.json'
    write_json(src,cache); train(src,out); production=read_json(out)
    assert reference.selection==production['selection']
    assert reference.relative_cost==production['pooled_relative_cost']
    xs=[r['features'] for r in cache['records']]
    assert reference.model.predict(xs).tolist()==[tree_route(production['tree'],x) for x in xs]


def test_inference_runs_detector_once():
    policy=fit_reference_policy(fixture_cache()); cfg=read_json('configs/default.json')
    calls=[]
    def detector(image):
        calls.append(image.shape)
        return [[0,.9,0.,0.,10.,10.]]
    route,pred=infer_one(np.full((40,50,3),30,np.uint8),policy,Enhancer(cfg),detector)
    assert route in ROUTES and len(calls)==1 and len(pred)==1


def test_rejects_test_and_overlap():
    cache=fixture_cache(); bad=copy.deepcopy(cache); bad['records'][0]['split']='test'
    with pytest.raises(ValueError,match='test data'): fit_reference_policy(bad)
    bad=copy.deepcopy(cache); bad['records'][-1]['sha256']=bad['records'][0]['sha256']
    with pytest.raises(ValueError,match='overlaps'): fit_reference_policy(bad)
