import pytest
from paired_ap_comparison import evaluator,resampled_ap
from selight.metrics import summary

def test_resampled_ap_matches_explicit_coco_recomputation():
    rows=[{'id':'a','width':100,'height':100,'boxes':[[0,0,0,20,20]]},
          {'id':'b','width':100,'height':100,'boxes':[[0,0,0,20,20]]},
          {'id':'c','width':100,'height':100,'boxes':[]}]
    predictions={'a':[[0,.9,0,0,20,20]],'b':[],'c':[[0,.95,0,0,20,20]]}
    ev=evaluator(rows,predictions)
    for indices in [[0,0,1],[2,0,1],[1,0,1,2]]:
        explicit=[{**rows[j],'id':str(i)} for i,j in enumerate(indices)]
        pred={str(i):predictions[rows[j]['id']] for i,j in enumerate(indices)}
        assert resampled_ap(ev,indices)==pytest.approx(summary(explicit,pred)['map_50_95'],abs=1e-12)

def test_no_annotation_resample_is_undefined():
    r=[{'id':'x','width':20,'height':20,'boxes':[]}]
    assert resampled_ap(evaluator(r,{'x':[]}),[0]) is None
