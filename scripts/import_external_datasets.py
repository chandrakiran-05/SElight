"""Import a fixed LLVIP visible test pilot and the official Penn-Fudan archive.

LLVIP ZIP byte ranges avoid downloading infrared/train images. No fitting.
Penn-Fudan download uses Windows system certificate trust through PowerShell.
"""
import argparse
import concurrent.futures
import hashlib
import io
import json
import struct
import subprocess
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
import zlib
from pathlib import Path

import cv2
import numpy as np
from selight.core import sha, write_json

REVISION = 'e50976add2fcca00b0db0f3bba59ed486d8c1f6d'
LLVIP_URL = f'https://huggingface.co/datasets/McCheng/LLVIP/resolve/{REVISION}/LLVIP.zip'
PENN_URL = 'https://www.cis.upenn.edu/~jshi/ped_html/PennFudanPed.zip'


def fetch(url, start=None, end=None):
    for attempt in range(4):
        try:
            headers = {} if start is None else {'Range': f'bytes={start}-{end}'}
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=90) as r:
                if start is not None and (r.status != 206 or not r.headers.get('Content-Range', '').startswith(f'bytes {start}-')):
                    raise ValueError('Server did not honor requested byte range')
                return r.read()
        except Exception:
            if attempt == 3:
                raise


class RemoteZip(io.RawIOBase):
    def __init__(self, url, size):
        self.url, self.size, self.pos = url, size, 0

    def seek(self, offset, whence=0):
        self.pos = offset if whence == 0 else self.pos + offset if whence == 1 else self.size + offset
        return self.pos

    def tell(self):
        return self.pos

    def read(self, n=-1):
        n = min(self.size-self.pos, n if n >= 0 else self.size-self.pos)
        if n <= 0:
            return b''
        b = fetch(self.url, self.pos, self.pos+n-1)
        self.pos += len(b)
        return b


def member(info):
    b = fetch(LLVIP_URL, info.header_offset, info.header_offset + 30 + len(info.filename.encode('utf-8')) + len(info.extra) + info.compress_size + 512)
    if b[:4] != b'PK\x03\x04':
        raise ValueError('Invalid local ZIP header')
    name_len, extra_len = struct.unpack_from('<HH', b, 26)
    start = 30 + name_len + extra_len
    packed = b[start:start+info.compress_size]
    result = zlib.decompress(packed, -15) if info.compress_type == zipfile.ZIP_DEFLATED else packed
    if len(result) != info.file_size or zlib.crc32(result) != info.CRC:
        raise ValueError('ZIP length/CRC mismatch')
    return result


def voc_boxes(payload, width, height):
    root = ET.fromstring(payload)
    if (int(root.findtext('size/width')), int(root.findtext('size/height'))) != (width, height):
        raise ValueError('Annotation/image dimensions differ')
    boxes = []
    for obj in root.findall('object'):
        if obj.findtext('name') != 'person' or int(obj.findtext('difficult', '0')) != 0:
            raise ValueError('Unexpected category or difficult annotation')
        # PASCAL VOC 1-based inclusive endpoints -> 0-based half-open pixel edges.
        xy = [float(obj.findtext('bndbox/'+k)) for k in ('xmin','ymin','xmax','ymax')]
        xy[0] -= 1; xy[1] -= 1
        if not (0 <= xy[0] < xy[2] <= width and 0 <= xy[1] < xy[3] <= height):
            raise ValueError('Invalid VOC box')
        boxes.append([0, *xy])
    return boxes


def record(dataset, path, boxes, image):
    return dict(id=dataset+'/'+path.stem, path=path.name, width=image.shape[1], height=image.shape[0], boxes=boxes, sha256=sha(path))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--limit', type=int, default=200)
    p.add_argument('--out', default='data/external')
    a = p.parse_args()
    if a.limit < 1:
        p.error('limit must be positive')
    out = Path(a.out).resolve(); out.mkdir(parents=True, exist_ok=True)
    ll = out/'llvip_visible_test'; ll.mkdir(exist_ok=True)
    with zipfile.ZipFile(RemoteZip(LLVIP_URL, 4003858172)) as z:
        infos = z.infolist()
    write_json(out/'llvip_archive_index.json', [dict(path=x.filename,size=x.file_size,crc=x.CRC) for x in infos])
    candidates = [x for x in infos if '/visible/test/' in '/'+x.filename and x.filename.lower().endswith('.jpg')]
    candidates.sort(key=lambda x: hashlib.sha256(x.filename.encode()).hexdigest())
    if len(candidates) < a.limit:
        raise ValueError('Insufficient test images')
    annotations = {Path(x.filename).stem:x for x in infos if '/Annotations/' in '/'+x.filename and x.filename.endswith('.xml')}

    def download(info):
        path = ll/Path(info.filename).name
        xml_path = path.with_suffix('.xml')
        if not path.exists(): path.write_bytes(member(info))
        if not xml_path.exists(): xml_path.write_bytes(member(annotations[path.stem]))
        image = cv2.imread(str(path))
        if image is None: raise ValueError('Cannot decode '+str(path))
        return record('llvip', path, voc_boxes(xml_path.read_bytes(), image.shape[1], image.shape[0]), image)

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        rows = list(pool.map(download, candidates[:a.limit]))
    write_json(ll/'manifest.json', dict(root=str(ll),records=rows,evaluation_classes=[0],kind='external_transfer',source=LLVIP_URL,revision=REVISION,source_split='official test (visible only)',available_images=len(candidates),selection='First N visible/test archive paths sorted by SHA256(path), before inference',annotation_conversion='VOC 1-based inclusive -> 0-based half-open; difficult unsupported',annotation_release='HF mirror; correspondence to 2023 official corrected annotation release unverified',license='Original LLVIP noncommercial research terms: https://github.com/bupt-ai-cz/LLVIP',fit_performed=False))
    print('LLVIP imported:',len(rows),flush=True)
    archive = out/'PennFudanPed.zip'
    if not archive.exists():
        subprocess.run(['powershell','-NoProfile','-Command',"Invoke-WebRequest -Uri '"+PENN_URL+"' -OutFile '"+str(archive).replace("'","''")+"' -UseBasicParsing"],check=True)
    penn = out/'penn_fudan'; penn.mkdir(exist_ok=True)
    rows = []
    with zipfile.ZipFile(archive) as z:
        for note in ('readme.txt','added-object-list.txt'):
            (penn/note).write_bytes(z.read('PennFudanPed/'+note))
        images = sorted(n for n in z.namelist() if '/PNGImages/' in n and n.endswith('.png'))
        for name in images:
            path = penn/Path(name).name; path.write_bytes(z.read(name))
            image = cv2.imread(str(path))
            mask_name = name.replace('/PNGImages/','/PedMasks/').replace('.png','_mask.png')
            mask = cv2.imdecode(np.frombuffer(z.read(mask_name),np.uint8),cv2.IMREAD_UNCHANGED)
            boxes = []
            for label in np.unique(mask):
                if label == 0: continue
                ys,xs = np.where(mask == label)
                boxes.append([0,int(xs.min()),int(ys.min()),int(xs.max())+1,int(ys.max())+1])
            rows.append(record('penn_fudan',path,boxes,image))
    write_json(penn/'manifest.json',dict(root=str(penn),records=rows,evaluation_classes=[0],kind='external_transfer',source=PENN_URL,source_split='all 170 images; no official held-out split used',archive_sha256=sha(archive),available_images=len(rows),selection='All released PNGImages',annotation_conversion='Human instance masks -> tight 0-based half-open boxes, maxima +1',domain='daylight pedestrian control, not low-light evidence',fit_performed=False))
    print('Penn-Fudan imported:',len(rows),flush=True)


if __name__ == '__main__':
    main()
