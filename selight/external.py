"""Explicit adaptation of official Zero-DCE++ with its pretrained checkpoint.

The official test script crops to multiples of 12. This adapter instead replicates
the border to the next multiple and crops the OUTPUT to preserve every GT object.
It is named zero_dcepp_scale12_pad and is not an exact official test reproduction.
"""
import importlib.util
from pathlib import Path
import numpy as np
import cv2
from .core import read_json, sha

class ZeroDCEPP:
    name='zero_dcepp_scale12_pad'
    def __init__(self,device,root='third_party/zero_dcepp'):
        import torch
        self.torch=torch; self.device='cpu' if device=='cpu' else f'cuda:{device}'
        root=Path(root); self.provenance=read_json(root/'provenance.json')
        for name in ['model.py','Epoch99.pth']:
            if sha(root/name)!=self.provenance['files'][name]['sha256']: raise ValueError('External asset hash mismatch')
        spec=importlib.util.spec_from_file_location('official_zero_dcepp',root/'model.py')
        module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        self.model=module.enhance_net_nopool(12).to(self.device).eval()
        self.model.load_state_dict(torch.load(root/'Epoch99.pth',map_location=self.device,weights_only=True),strict=True)
    def apply(self,bgr):
        t=self.torch; h,w=bgr.shape[:2]
        rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
        x=t.from_numpy(np.ascontiguousarray(rgb.transpose(2,0,1))).to(self.device,dtype=t.float32).unsqueeze(0)/255.
        x=t.nn.functional.pad(x,(0,(-w)%12,0,(-h)%12),mode='replicate')
        with t.inference_mode(): enhanced,_=self.model(x)
        # Quantization follows torchvision save_image's multiply/add/clamp convention.
        array=enhanced[0,:,:h,:w].permute(1,2,0).mul(255).add(.5).clamp(0,255).to(t.uint8).cpu().numpy()
        return cv2.cvtColor(array,cv2.COLOR_RGB2BGR)
