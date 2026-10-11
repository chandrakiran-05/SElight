"""Controlled resolution/noise/JPEG stress test of a FROZEN SELight policy.

This simulates limited RGB inputs. It does not simulate a complete camera sensor
or prove performance on inexpensive cameras. Test outputs never train a policy.
"""
import argparse
import csv
import hashlib
import time
from pathlib import Path
import cv2
import numpy as np
from selight.core import (ROUTES, Enhancer, select, load_image, read_json,
                          write_json, sha)
from selight.data import audit
from selight.experiment import signature, environment, records_for
from selight.metrics import summary

CONDITIONS = {
    'native': {'max_side': 0, 'noise_sigma': 0., 'jpeg_quality': None},
    'edge640': {'max_side': 640, 'noise_sigma': 0., 'jpeg_quality': None},
    'edge320': {'max_side': 320, 'noise_sigma': 0., 'jpeg_quality': None},
    'edge320_noise5_jpeg50': {'max_side': 320, 'noise_sigma': 5., 'jpeg_quality': 50},
}


def simulate_input(image, condition, image_id, seed=42):
    """Deterministic stress input, same pixels for every compared method.

    Noise is additive RGB-space Gaussian noise, NOT calibrated sensor noise.
    Return image plus exact x/y ratios for original-coordinate restoration.
    """
    h, w = image.shape[:2]
    edge = condition['max_side']
    sigma = condition['noise_sigma']
    jpeg = condition['jpeg_quality']
    if edge < 0 or sigma < 0 or (jpeg is not None and not 1 <= jpeg <= 100):
        raise ValueError('Invalid stress condition')
    scale = min(1., edge / max(h, w)) if edge else 1.
    nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
    out = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_AREA) if (nw, nh) != (w, h) else image.copy()
    if sigma:
        key = hashlib.sha256(f'{seed}:{image_id}'.encode()).digest()
        rng = np.random.default_rng(int.from_bytes(key[:8], 'big'))
        out = np.clip(np.rint(out.astype(np.float32) + rng.normal(0, sigma, out.shape)), 0, 255).astype(np.uint8)
    if jpeg is not None:
        ok, encoded = cv2.imencode('.jpg', out, [cv2.IMWRITE_JPEG_QUALITY, jpeg])
        if not ok:
            raise RuntimeError('JPEG simulation failed')
        out = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if out is None:
            raise RuntimeError('JPEG decoding failed')
    return out, (nw / w, nh / h)


def restore_boxes(predictions, ratios):
    sx, sy = ratios
    return [[c, score, x1/sx, y1/sy, x2/sx, y2/sy]
            for c, score, x1, y1, x2, y2 in predictions]


def run(manifest, config, policy_path, output, split='val', repeats=3, limit=None,
        condition_names=None):
    from selight.detector import Detector
    obj, cfg, policy = read_json(manifest), read_json(config), read_json(policy_path)
    audit(obj)
    rows = records_for(obj, split, limit)
    if not rows:
        raise ValueError('Selected split is empty')
    if repeats < 1:
        raise ValueError('repeats must be positive')
    if policy['training_manifest_hash'] != sha(manifest):
        raise ValueError('Manifest differs from training manifest')
    if split == 'test' and {r['sha256'] for r in rows} & set(policy['training_hashes'] + policy['validation_hashes']):
        raise ValueError('Held-out content overlaps development')
    names = condition_names or list(CONDITIONS)
    if len(names) != len(set(names)) or any(n not in CONDITIONS for n in names):
        raise ValueError('Unknown or repeated condition')
    det, enh = Detector(cfg), Enhancer(cfg)
    if signature(cfg, det.weight_hash) != policy['signature']:
        raise ValueError('Policy/config/checkpoint mismatch')
    out = Path(output); out.mkdir(parents=True, exist_ok=False)
    cv2.setNumThreads(cfg['threads'])
    rng = np.random.default_rng(cfg['seed'])
    methods = (*ROUTES, 'threshold', 'selector') + tuple(a for a,key in [('brightness_tree','brightness_tree'),('best_fixed','best_fixed_route')] if key in policy)
    logs, metrics = [], {}
    for condition_name in names:
        condition = CONDITIONS[condition_name]
        warm_image, _ = simulate_input(load_image(Path(obj['root'])/rows[0]['path']), condition, rows[0]['id'])
        # Warm all methods including the selector before timing this condition.
        for method in methods:
            for _ in range(10):
                route = select(warm_image, method, cfg, policy)
                det.predict(enh.apply(warm_image, route))
        det.sync()
        for rep in range(repeats):
            for method in rng.permutation(methods):
                method = str(method); predictions = {}; times = []; counts = dict.fromkeys(ROUTES, 0)
                for record in rows:
                    # Decode and create synthetic input OUTSIDE the timed scope.
                    # These operations simulate the input source, not the method.
                    source = load_image(Path(obj['root'])/record['path'])
                    image, ratios = simulate_input(source, condition, record['id'], cfg['seed'])
                    det.sync(); started = time.perf_counter()
                    route = select(image, method, cfg, policy); selected = time.perf_counter()
                    processed = enh.apply(image, route); enhanced = time.perf_counter()
                    pred = det.predict(processed); det.sync()
                    pred = restore_boxes(pred, ratios); finished = time.perf_counter()
                    predictions[record['id']] = pred; counts[route] += 1
                    elapsed = (finished-started)*1000; times.append(elapsed)
                    logs.append(dict(condition=condition_name, repeat=rep, method=method,
                                     id=record['id'], route=route, input_width=image.shape[1],
                                     input_height=image.shape[0], selector_ms=(selected-started)*1000,
                                     enhancement_ms=(enhanced-selected)*1000,
                                     detector_and_coordinates_ms=(finished-enhanced)*1000,
                                     processing_ms=elapsed))
                key = f'{condition_name}__{method}__repeat{rep}'
                result = summary(rows, predictions, cfg['operating_confidence'])
                result.update(mean_processing_ms=float(np.mean(times)), p95_processing_ms=float(np.percentile(times, 95)),
                              serial_processing_fps=1000/float(np.mean(times)), route_counts=counts)
                metrics[key] = result
                write_json(out/f'{key}_predictions.json', predictions)
    with (out/'timings.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(logs[0])); writer.writeheader(); writer.writerows(logs)
    write_json(out/'summary.json', dict(results=metrics, conditions={n:CONDITIONS[n] for n in names},
        split=split, pilot_limit=limit, environment=environment(cfg), policy_hash=sha(policy_path),
        manifest_hash=sha(manifest), weights_hash=det.weight_hash,
        policy_scope='Unmodified frozen policy transferred to declared synthetic input conditions',
        timing_scope='Processing only: feature extraction, routing, enhancement, detector and coordinate restoration; excludes source decode, stress simulation, acquisition, rendering and output writing',
        accuracy_scope='COCO-style AP in ORIGINAL coordinates; AP-small uses original annotation area',
        limitations='Synthetic RGB stress is not real camera validation; no calibrated lux or sensor model'))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', required=True); p.add_argument('--config', default='configs/default.json')
    p.add_argument('--policy', required=True); p.add_argument('--out', required=True)
    p.add_argument('--split', choices=['val','test'], default='val')
    p.add_argument('--repeats', type=int, default=3); p.add_argument('--limit', type=int)
    p.add_argument('--conditions', nargs='+', choices=list(CONDITIONS))
    a = p.parse_args()
    if a.limit is not None and a.limit < 1: p.error('limit must be positive')
    run(a.manifest, a.config, a.policy, a.out, a.split, a.repeats, a.limit, a.conditions)

if __name__ == '__main__':
    main()
