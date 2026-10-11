"""Readable reference for the existing SELight algorithm (not a new model).

Run from the project root. Reuses exactly the production image features and
transforms. Reference training matches experiment.train's selector search;
use the normal CLI to export a complete policy with hashes/provenance.
"""
from dataclasses import dataclass
from typing import Callable
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from selight.core import ROUTES, Enhancer, features


@dataclass
class ReferencePolicy:
    model: DecisionTreeClassifier
    relative_cost: dict
    selection: dict


def choose_training_route(quality: dict, cost: dict,
                          cost_weight: float, margin: float = 0.02) -> str:
    """Use annotated DEVELOPMENT images only; never use this at inference.

    quality[r]: per-image F1 at IoU >= 0.5 and confidence >= 0.25.
    cost[r]: pooled relative extra route time, original cost is zero.
    Ties prefer original, then gamma, then CLAHE.
    """
    if cost_weight < 0 or margin < 0:
        raise ValueError('Cost weight and margin must be nonnegative.')
    for route in ROUTES:
        if not np.isfinite(quality[route]) or not 0 <= quality[route] <= 1:
            raise ValueError('Quality must be finite and in [0, 1].')
        if not np.isfinite(cost[route]) or cost[route] < 0:
            raise ValueError('Cost must be finite and nonnegative.')
    if cost['original'] != 0:
        raise ValueError('Original must have zero extra cost.')

    best_route = 'original'
    best_utility = quality['original']
    for route in ('gamma', 'clahe'):
        # Use the exact comparison used by the production implementation.
        if quality[route] < quality['original'] + margin:
            continue
        utility = quality[route] - cost_weight * cost[route]
        if utility > best_utility:  # Strict > preserves deterministic tie order.
            best_route, best_utility = route, utility
    return best_route


def fit_reference_policy(cache: dict) -> ReferencePolicy:
    """Match the existing train() candidate search, without writing files.

    Expects the JSON produced by `python -m selight cache`.
    The production CLI additionally writes policy.json, readable rules,
    threshold-baseline selection, oracle diagnostics and provenance hashes.
    """
    rows = cache['records']
    if any(r['split'] not in ('train', 'val') for r in rows):
        raise ValueError('Development cache must not contain test data.')
    train = [r for r in rows if r['split'] == 'train']
    val = [r for r in rows if r['split'] == 'val']
    if not train or not val:
        raise ValueError('Nonempty train and validation sets are required.')
    if {r['sha256'] for r in train} & {r['sha256'] for r in val}:
        raise ValueError('Training and validation content overlaps.')
    for row in rows:
        if len(row['features']) != 8 or not np.isfinite(row['features']).all():
            raise ValueError('Each image needs eight finite features.')
        if any(not np.isfinite(row['ms'][r]) or row['ms'][r] <= 0 for r in ROUTES):
            raise ValueError('Timing samples must be finite and positive.')

    base_ms = float(np.median([r['ms']['original'] for r in train]))
    cost = {
        route: max(0.0, float(np.median([r['ms'][route] for r in train])) / base_ms - 1.0)
        for route in ROUTES
    }
    cost['original'] = 0.0
    x_train = [r['features'] for r in train]
    x_val = [r['features'] for r in val]
    best = None

    # Same grid/order as production. These are declared starting settings,
    # not universal constants or parameters selected using test results.
    for depth in (2, 3, 4):
        for training_lambda in (0.0, 0.01, 0.05):
            targets = [choose_training_route(r['quality'], cost, training_lambda)
                       for r in train]
            tree = DecisionTreeClassifier(
                max_depth=depth,
                min_samples_leaf=max(2, int(0.02 * len(train))),
                random_state=42,
            )
            tree.fit(x_train, targets)
            selected = tree.predict(x_val)
            mean_f1 = float(np.mean([r['quality'][a] for r, a in zip(val, selected)]))
            mean_cost = float(np.mean([cost[a] for a in selected]))
            # FIXED validation weight makes candidate objectives comparable.
            objective = mean_f1 - 0.01 * mean_cost
            settings = dict(depth=depth, training_lambda=training_lambda,
                            val_mean_image_f1=mean_f1, val_extra_cost=mean_cost,
                            val_objective=objective)
            if best is None or objective > best.selection['val_objective'] + 1e-12:
                best = ReferencePolicy(tree, cost, settings)
    return best


def infer_one(image: np.ndarray, policy: ReferencePolicy,
              enhancer: Enhancer, detector_predict: Callable,
              feature_size: int = 64):
    """One selected transform and exactly one detector call; no labels needed.

    Construct the enhancer and detector once, outside the image loop.
    Input is OpenCV BGR uint8. Callback is normally Detector.predict.
    """
    x = np.asarray([features(image, feature_size)], dtype=np.float32)
    route = str(policy.model.predict(x)[0])
    processed = enhancer.apply(image, route)
    detections = detector_predict(processed)
    return route, detections
