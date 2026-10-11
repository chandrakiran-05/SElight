"""Fetch pinned official research assets; no substituted architecture/checkpoint."""
import urllib.request
import urllib.parse
import json
import hashlib
from pathlib import Path

root=Path('third_party/zero_dcepp'); root.mkdir(parents=True,exist_ok=True)
req=urllib.request.Request('https://api.github.com/repos/Li-Chongyi/Zero-DCE_extension/commits/main',headers={'User-Agent':'SELight-research'})
commit=json.load(urllib.request.urlopen(req,timeout=30))['sha']
files={}
for local,remote in [('model.py','Zero-DCE++/model.py'),('lowlight_test.py.reference','Zero-DCE++/lowlight_test.py'),('Epoch99.pth','Zero-DCE++/snapshots_Zero_DCE++/Epoch99.pth'),('README.md','README.md')]:
    url='https://raw.githubusercontent.com/Li-Chongyi/Zero-DCE_extension/'+commit+'/'+urllib.parse.quote(remote,safe='/')
    data=urllib.request.urlopen(url,timeout=30).read(); (root/local).write_bytes(data)
    files[local]={'url':url,'sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data)}
(root/'provenance.json').write_text(json.dumps({'commit':commit,'files':files,'licence':'Academic research; CC BY-NC 4.0 per official README'},indent=2),encoding='utf-8')
print(json.dumps({'commit':commit,'files':files},indent=2))
