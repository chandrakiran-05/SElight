"""Established transforms and the proposed feature-to-route selector."""
import hashlib
import json
from pathlib import Path
import cv2
import numpy as np

ROUTES = ("original", "gamma", "clahe")
FEATURES = ("mean", "std", "p10", "p50", "p90", "dark_fraction", "bright_fraction", "gradient")
COCO_IDS = (0, 1, 2, 3, 5)
NAMES = ("person", "bicycle", "car", "motorcycle", "bus")

def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1048576), b""):
            h.update(b)
    return h.hexdigest()

def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()

def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))

def write_json(path, obj):
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, allow_nan=False), encoding="utf-8")

def load_image(path):
    # imdecode handles Unicode Windows paths.
    im = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if im is None:
        raise ValueError(f"Cannot decode image: {path}")
    return im

class Enhancer:
    def __init__(self, cfg):
        self.cfg = cfg
        if cfg["gamma"] <= 0:
            raise ValueError("gamma must be positive")
        # Explicit convention y=x**gamma; gamma below 1 brightens.
        self.lut = np.round(255*(np.arange(256)/255.)**cfg["gamma"]).astype(np.uint8)
        g = cfg["clahe_grid"]
        self.clahe = cv2.createCLAHE(clipLimit=cfg["clahe_clip"], tileGridSize=(g, g))

    def apply(self, image, route):
        if route == "original": return image
        if route == "gamma": return cv2.LUT(image, self.lut)
        if route == "clahe":
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            lab[:, :, 0] = self.clahe.apply(lab[:, :, 0])
            return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        raise ValueError(f"Unknown route: {route}")

def features(image, size=64):
    # Resize before grayscale so feature work is independent of detector resolution.
    small = cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)
    y = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32)/255.
    p = np.percentile(y, [10, 50, 90])
    grad = (np.abs(np.diff(y, axis=0)).mean()+np.abs(np.diff(y, axis=1)).mean())/2
    return [float(v) for v in [y.mean(), y.std(), *p, (y < .2).mean(), (y > .95).mean(), grad]]

def tree_route(tree, x):
    x = np.asarray(x, dtype=np.float32)  # Match scikit-learn prediction precision.
    node = 0
    while tree["left"][node] != -1:
        node = tree["left"][node] if x[tree["feature"][node]] <= tree["threshold"][node] else tree["right"][node]
    return tree["route"][node]

def select(image, method, cfg, policy=None):
    if method in ROUTES: return method
    if method == "best_fixed" and policy:
        return policy["best_fixed_route"]
    x = features(image, cfg["feature_size"])
    if method == "threshold":
        threshold = policy.get("brightness_threshold", cfg["brightness_threshold"]) if policy else cfg["brightness_threshold"]
        return "gamma" if x[0] < threshold else "original"
    if method == "selector" and policy:
        return tree_route(policy["tree"], x[:1] if policy.get('ablation')=='brightness_only' else x)
    if method == "brightness_tree" and policy:
        return tree_route(policy["brightness_tree"], [x[0]])
    raise ValueError("selector requires a trained policy")
