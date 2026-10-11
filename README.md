# SELight — current codebase

The new research package is in [`research/`](research/): [manuscript](research/PAPER.md), [complete context](research/RESEARCH_CONTEXT.md), [code tree](research/CODE_MAP.md), [code review](research/CODE_REVIEW.md), [runbook](research/RUNBOOK.md), and [visual showcase](research/showcase.html). It includes a fresh matched five-method comparison on 200 validation images, raw predictions, paired intervals and exportable figures. The policy remains frozen and currently matches the original detector.

Detect person, bicycle, car, motorcycle and bus in ordinary RGB photos/videos using frozen YOLOv8n and a small enhancement-route selector. The saved selector currently always chooses original input; enhancement gains have not been demonstrated.

## 1. Check the installation

Open PowerShell in this folder. The existing `.venv` already works; no installation or activation is needed.

```powershell
cd "E:\first paper\02_SELight_Project_Antigravity\selight_project"
.\.venv\Scripts\python.exe -m selight doctor
.\.venv\Scripts\python.exe -m pytest -q
```

Expected: `CUDA available: True`, RTX 4050 Laptop GPU, and **31 passing tests**. Tests check implementation correctness; they do not establish real-world accuracy.

## 2. Run on your own photo

Replace the input path with your image. Quote paths containing spaces.

```powershell
.\.venv\Scripts\python.exe infer_photo.py --image "C:\path\night_photo.jpg" --config configs/gpu.json --out outputs/photo
```

Open `outputs/photo/annotated.png` to inspect boxes, labels and confidence. `outputs/photo/detections.json` contains raw predictions collected at confidence .001; the image displays predictions at confidence .25. The default trained policy is `models/policy.json`.

The output folder must be new. For another image, use `outputs/photo2` or a different name.

## 3. Run on your own video

```powershell
.\.venv\Scripts\python.exe -m selight demo --source "C:\path\night_video.mp4" --config configs/gpu.json --out outputs/video/annotated.mp4
```

Open the output MP4 in a video player. Frames are processed independently. Video playback FPS comes from the source and is not inference speed. `--source 0` opens the laptop's default webcam; stop the process with Ctrl+C.

For CPU execution, replace `configs/gpu.json` with `configs/cpu.json` in either command.

## 4. Immediately usable sample commands

The archived dataset and slideshow remain available outside the clean code folder. These commands work from this folder without downloading anything:

```powershell
.\.venv\Scripts\python.exe infer_photo.py --image "..\selight_project_backup_20261002\data\raw\ExDark_extracted\ExDark\Bicycle\2015_00375.jpg" --config configs/gpu.json --out outputs/sample_photo
.\.venv\Scripts\python.exe -m selight demo --source "..\selight_project_backup_20261002\data\demo\night_clip.mp4" --config configs/gpu.json --out outputs/sample_video/annotated.mp4
```

The sample photo includes a known class error (bicycle predicted as motorcycle). The sample video is a dataset slideshow, not genuine phone footage. Use your own recordings to check actual camera behavior.

## Current folder contents

| Path | Purpose |
|---|---|
| `selight/` | Features, transforms, selector training, frozen detector, metrics, data conversion and CLI |
| `configs/` | CPU/GPU detector and enhancement settings |
| `models/policy.json` | Trained selector with provenance |
| `models/policy.rules.txt` | Readable learned rules |
| `models/yolov8n.pt` | Frozen detector checkpoint |
| `tests/` | 31 correctness checks |
| `infer_photo.py` | Photo detection and annotation |
| `camera_benchmark.py` | Labelled resolution/noise/JPEG stress experiments |
| `video_benchmark.py` | Local-file processing timing, without drawing/encoding |
| `algorithm_reference.py` | Readable equivalent selector implementation |
| `paired_comparison.py` | Paired mean per-image F1 bootstrap |
| `paired_ap_comparison.py` | Paired COCO AP recomputation bootstrap, with optional scene clusters |
| `scripts/` | Reusable ablation, visual-audit and external-baseline tools |
| `third_party/zero_dcepp/` | Official enhancement-baseline source/checkpoint and attribution |
| `.venv/` | Existing working Python environment |
| `requirements*.txt`, `environment-lock.txt` | Dependencies and the verified environment lock |

Old datasets, outputs, manuscripts, source backups, ZIPs and one-off review scripts were moved to:

`E:\first paper\02_SELight_Project_Antigravity\selight_project_backup_20261002`

They were removed from the active code folder without permanent deletion. Archived manifests contain their original dataset-root paths; restore the original data location or create a newly audited manifest before continuing labelled experiments. Do not bypass manifest/policy checks.

## Research commands

Use `--help` to see every argument:

```powershell
.\.venv\Scripts\python.exe -m selight --help
.\.venv\Scripts\python.exe camera_benchmark.py --help
.\.venv\Scripts\python.exe -m scripts.development_ablations --help
.\.venv\Scripts\python.exe -m scripts.visual_audit --help
.\.venv\Scripts\python.exe -m scripts.external_pilot --help
```

Training/evaluation requires a labelled dataset manifest and matching cache/policy provenance. Photo/video inference only needs the included weights and saved policy. Changing detector settings such as `imgsz` requires a compatible retrained policy; switching CPU/GPU does not.

Keep the existing environment. `setup_windows.ps1` preserves an existing `.venv` and is only for a fresh setup. See `THIRD_PARTY_NOTICES.md` for external model attribution.

External human-labelled results are included in `research/PAPER.md`, `research/external_comparison.csv`, and `research/showcase.html`. The Hugging Face LLVIP run is a 200-image visible test pilot; official Penn-Fudan uses all 170 images as a daylight control. Scores are person-only and the current selector equals original input. Reproduction commands are in `research/RUNBOOK.md`.
