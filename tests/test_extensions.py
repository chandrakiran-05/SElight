import numpy as np
import pytest
from selight.core import read_json, write_json, select
from selight.experiment import train
from selight.data import prepare

def make_cache():
    rows=[]
    for split in ['train','val']:
        for i in range(36):
            target=['original','gamma','clahe'][i%3]
            rows.append(dict(id=f'{split}{i}',sha256=f'{split}{i}',split=split,features=[i%3/3]+[0]*7,
                quality={a:.9 if a==target else .3 for a in ['original','gamma','clahe']},ms={'original':10,'gamma':11,'clahe':12}))
    return dict(signature='fixture',manifest_hash='fixture',environment={'config':{'device':'cpu'}},records=rows)

@pytest.mark.parametrize('ablation',['full','no_cost','no_margin','brightness_only','no_clahe'])
def test_ablation_semantics(tmp_path,ablation):
    src=tmp_path/'cache.json'; dest=tmp_path/'policy.json'; write_json(src,make_cache()); train(src,dest,ablation)
    p=read_json(dest)
    assert p['validation_cost_weight']==(0 if ablation=='no_cost' else .01)
    assert p['gain_margin']==(0 if ablation=='no_margin' else .02)
    assert len(p['feature_names'])==(1 if ablation=='brightness_only' else 8)
    if ablation=='no_clahe': assert 'clahe' not in p['tree']['route'] and 'clahe' not in p['brightness_tree']['route']
    if ablation=='no_cost': assert all(c['training_lambda']==0 for c in p['candidates'])
    assert set(p['brightness_tree']['route'])==({'original','gamma'} if ablation=='no_clahe' else {'original','gamma','clahe'})
    cfg=read_json('configs/default.json')
    assert select(np.zeros((20,30,3),np.uint8),'best_fixed',cfg,p)==p['best_fixed_route']

def test_prepare_requires_explicit_documented_deduplication(tmp_path):
    import cv2
    ims=tmp_path/'ims'; anns=tmp_path/'anns'; ims.mkdir(); anns.mkdir()
    for name,value in [('a',10),('b',10),('v',30),('t',50)]:
        cv2.imwrite(str(ims/f'{name}.png'),np.full((20,30,3),value,np.uint8))
        (anns/f'{name}.png.txt').write_text('People 1 1 4 5')
    sf=tmp_path/'split.txt'; sf.write_text('a.png 11 1 1 1\nb.png 11 1 1 3\nv.png 11 1 1 2\nt.png 11 1 1 3')
    out=tmp_path/'manifest.json'
    with pytest.raises(ValueError,match='Exact duplicate'): prepare(ims,anns,sf,out,0)
    prepare(ims,anns,sf,out,0,True)
    p=read_json(out); assert p['deduplication']['omitted'][0]['omitted']=='b.png'
    assert 'modified split' in p['source'] and len(p['records'])==3
