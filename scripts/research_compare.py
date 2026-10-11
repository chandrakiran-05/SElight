"""Matched validation comparison of a frozen detector and five input methods.

No fitting or test inference. Decode, startup, metrics, drawing and writing are
outside the synchronized processing timer. Raw repetitions remain available.
"""
import argparse
import collections
import contextlib
import csv
import io
import time
from pathlib import Path

import cv2
import numpy as np
from selight.core import Enhancer, load_image, read_json, select, sha, write_json
from selight.data import audit
from selight.detector import Detector
from selight.experiment import environment, records_for, signature
from selight.external import ZeroDCEPP
from selight.metrics import summary

METHODS = ('original', 'gamma', 'clahe', 'selector', 'zero_dcepp_scale12_pad')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', required=True)
    p.add_argument('--policy', default='models/policy.json')
    p.add_argument('--config', default='configs/gpu.json')
    p.add_argument('--out', required=True)
    p.add_argument('--limit', type=int, default=200)
    p.add_argument('--repeats', type=int, default=3)
    a = p.parse_args()
    if a.limit < 1 or a.repeats < 1:
        p.error('limit and repeats must be positive')
    obj, cfg, policy = read_json(a.manifest), read_json(a.config), read_json(a.policy)
    if policy['training_manifest_hash'] != sha(a.manifest):
        raise ValueError('Use the manifest matching this frozen policy')
    data_audit = audit(obj)
    rows = records_for(obj, 'val', a.limit)
    if len(rows) != a.limit:
        raise ValueError('Requested more validation images than available')
    det, enh, ext = Detector(cfg), Enhancer(cfg), ZeroDCEPP(cfg['device'])
    if signature(cfg, det.weight_hash) != policy['signature']:
        raise ValueError('Policy/config/checkpoint mismatch')
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    cv2.setNumThreads(cfg['threads'])
    write_json(out/'sample.json', {
        'records': rows, 'manifest_hash': sha(a.manifest),
        'selection': 'First limit validation IDs sorted by SHA256 of JSON ID, same as records_for',
        'split': 'val', 'fit_performed': False, 'independent_camera_sessions_verified': False})

    def prepare(image, method):
        if method == ext.name:
            return ext.apply(image), ext.name
        route = select(image, method, cfg, policy)
        return enh.apply(image, route), route

    first = load_image(Path(obj['root'])/rows[0]['path'])
    for method in METHODS:
        for _ in range(10):
            processed, _ = prepare(first, method)
            det.predict(processed)
    det.sync()
    rng = np.random.default_rng(cfg['seed'])
    results, logs, repetitions = {}, [], {}
    for rep in range(a.repeats):
        for method in rng.permutation(METHODS):
            method = str(method)
            predictions, routes = {}, []
            if cfg['device'] != 'cpu':
                det.torch.cuda.reset_peak_memory_stats()
            for index in rng.permutation(len(rows)):
                r = rows[int(index)]
                image = load_image(Path(obj['root'])/r['path'])
                if image.shape[:2] != (r['height'], r['width']):
                    raise ValueError('Manifest dimension mismatch: '+r['id'])
                det.sync()
                start = time.perf_counter()
                processed, route = prepare(image, method)
                det.sync()
                enhanced = time.perf_counter()
                pred = det.predict(processed)
                det.sync()
                end = time.perf_counter()
                predictions[r['id']] = pred
                routes.append(route)
                logs.append(dict(method=method, repeat=rep, id=r['id'], route=route,
                                 selection_enhancement_ms=(enhanced-start)*1000,
                                 detector_ms=(end-enhanced)*1000,
                                 processing_ms=(end-start)*1000))
            key = f'{method}_repeat{rep}'
            write_json(out/f'{key}_predictions.json', predictions)
            with contextlib.redirect_stdout(io.StringIO()):
                m = summary(rows, predictions, cfg['operating_confidence'])
            times = [x['processing_ms'] for x in logs if x['repeat']==rep and x['method']==method]
            m.update(mean_ms=float(np.mean(times)), p95_ms=float(np.percentile(times,95)),
                     median_ms=float(np.median(times)), route_counts=dict(collections.Counter(routes)),
                     cuda_peak_allocated_mib=det.torch.cuda.max_memory_allocated()/2**20 if cfg['device']!='cpu' else None)
            results[key] = m
            if method not in repetitions:
                repetitions[method] = predictions
            m['predictions_equal_repeat0'] = predictions == repetitions[method]
            print(f'{key}: AP={m["map_50_95"]:.5f}, F1={m["f1"]:.5f}, processing={m["mean_ms"]:.2f} ms', flush=True)
    with (out/'timings.csv').open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(logs[0]))
        w.writeheader(); w.writerows(logs)
    aggregate = {}
    for method in METHODS:
        measurements = [v for k,v in results.items() if k.startswith(method+'_repeat')]
        times = [x['processing_ms'] for x in logs if x['method']==method]
        aggregate[method] = {**measurements[0],
            'mean_ms': float(np.mean(times)), 'p95_ms': float(np.percentile(times,95)),
            'median_ms': float(np.median(times)),
            'repeat_means_ms': [v['mean_ms'] for v in measurements],
            'accuracy_repeats_identical': all(v['predictions_equal_repeat0'] for v in measurements),
            'accuracy_source': 'repeat0; timing repetitions are not independent accuracy samples'}
    write_json(out/'summary.json', dict(results=results, aggregate=aggregate,
        manifest_hash=sha(a.manifest), policy_hash=sha(a.policy), weights_hash=det.weight_hash,
        environment=environment(cfg), dataset_audit=data_audit, split='val', pilot_limit=a.limit,
        repeats=a.repeats, external_provenance=ext.provenance,
        selector_predictions_equal_original=repetitions['selector']==repetitions['original'],
        timing_scope='Synchronized processing: selection/enhancement, transfers, uint8 quantization, detector; excludes decode/startup/metrics/rendering/writing/acquisition',
        adaptation='Official Zero-DCE++ scale12; replicate-pad input, crop output; not official input-crop protocol',
        methodology='Same images, annotations, detector/checkpoint, FP32, thresholds; randomized method and image order; 10 warmups per method; no training or test inference'))
    print('Comparison completed: '+str(out), flush=True)


if __name__ == '__main__':
    main()
