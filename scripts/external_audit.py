"""Validate external human-labelled records without inventing training splits."""
import math
from pathlib import Path
from selight.core import sha, load_image


def audit_external(obj, reference):
    if obj['kind'] != 'external_transfer' or obj['evaluation_classes'] != [0]:
        raise ValueError('Only declared external person evaluation is supported')
    root = Path(obj['root']).resolve()
    reference_hashes = {r['sha256'] for r in reference['records']}
    ids, hashes = set(), set()
    for r in obj['records']:
        path = (root/r['path']).resolve()
        if not path.is_relative_to(root): raise ValueError('Path escapes dataset')
        h = sha(path)
        if h != r['sha256'] or h in hashes or h in reference_hashes:
            raise ValueError('Hash mismatch, exact duplicate or ExDark overlap')
        if r['id'] in ids: raise ValueError('Duplicate ID')
        if load_image(path).shape[:2] != (r['height'],r['width']):
            raise ValueError('Dimensions disagree')
        for c,x1,y1,x2,y2 in r['boxes']:
            if c != 0 or not all(math.isfinite(x) for x in (x1,y1,x2,y2)) or not (0 <= x1 < x2 <= r['width'] and 0 <= y1 < y2 <= r['height']):
                raise ValueError('Invalid person annotation')
        ids.add(r['id']); hashes.add(h)
    if not ids: raise ValueError('Empty evaluation')
    return dict(images=len(ids), boxes=sum(len(r['boxes']) for r in obj['records']), exact_duplicates=0, exact_exdark_overlap=0, pretraining_overlap_verified=False, session_independence_verified=False, fit_performed=False)
