# Complete SELight research context

Updated 12 October 2026 from local source, fresh predictions and preserved evidence. Start with `PAPER.md` for the manuscript, `showcase.html` for visual results, and `CODE_MAP.md` for the implementation.

A shareable updated PDF is in `../output/pdf/SELight_research_paper.pdf`. It was exported from the verified Markdown and visually checked. The editable standalone `paper.tex` is also preserved; native compilation did not return a result in this session and remains unverified.

## Research identity

**Working title:** SELight: An Audited Study of Enhancement Selection for Low-Light Object Detection with a Frozen Lightweight Detector.

**Question:** Can inexpensive, detection-aware input enhancement selection improve a fixed lightweight detector on low-light ordinary RGB images at an acceptable measured processing cost?

**Current answer:** The implemented selector has not shown an advantage. Its saved decision tree sends every image to the original-input route. Fresh and historical prediction checks confirm equality with plain YOLOv8n. This is a working, reproducible prototype with negative findings; a perfect or state-of-the-art research claim would be unsupported.

The user's intended application is laptop detection of person, bicycle, car, motorcycle and bus in night photos/videos from ordinary cameras. Outputs are boxes, labels and confidence. The project does not implement licence-plate reading, face recognition, autonomous driving, thermal inference, phone-side inference or reconstruction of details absent from the signal. The abandoned PRDASV/frame-skipping approach is not part of this direction.

## What is proposed and what is reused

YOLOv8n, CART, gamma correction, Lab CLAHE and Zero-DCE++ are established components. The investigated project idea is using an eight-feature, development-trained tree to choose one cheap transform before exactly one frozen-detector call. It is not a newly invented detector or image-enhancement network. Novelty beyond this implementation is not established by these experiments.

The detection-aware route target is computed from annotated development images. Available routes are original, gamma and CLAHE. A route must improve per-image F1 by at least .02 to be eligible; utility subtracts lambda times pooled relative extra processing cost. The CART search covers depth 2/3/4 and training lambda 0/.01/.05. Validation always uses a fixed .01 cost weight. Original wins ties. The selected model uses depth 2 and lambda 0; every leaf returns original.

The eight features are mean, standard deviation, grayscale p10/p50/p90, fraction below .2, fraction above .95 and mean finite-difference gradient magnitude on a 64x64 resized input. Original image geometry is retained for detection. Gamma uses exponent .65. CLAHE changes Lab luminance with clip limit 2 and an 8x8 grid.

## Fixed detector and environment

Frozen COCO YOLOv8n, input size 640, batch 1, FP32, AP collection threshold .001, displayed/F1 operating threshold .25, NMS IoU .7 and maxDet 100. Five-class mapping is documented in `CODE_MAP.md`. Policy signatures validate the actual checkpoint hash and detector/enhancer settings; device/thread/path changes are explicitly excluded from the signature.

The observed execution environment is Python 3.10.11, PyTorch 2.6.0+cu124, torchvision 0.21.0+cu124, Ultralytics 8.4.171, OpenCV 4.11.0.86, NumPy 2.2.6, scikit-learn 1.7.2 and pycocotools 2.0.11. CUDA is available on an RTX 4050 Laptop GPU. Earlier hardware inspection recorded i5-13450HX and 16 GB RAM. Software versions and source hashes are saved with the fresh run. The current test suite passes 31 tests.

## Data and independence

ExDark publishes 7,363 low-light RGB images and 12 annotation classes. This project evaluates only the five overlapping COCO classes and retains images without those target objects as negatives. It therefore reports a five-class subset result, not the complete 12-class ExDark benchmark.

The retained legacy manifest has 7,345 images: train 2,993 / validation 1,792 / test 2,560, after 18 exact duplicate removals. Negative counts are 1,219 / 785 / 1,278. The legacy manifest's old source string understates the modifications; this context records them explicitly. The prior reviewed sensitivity manifest removes six visually confirmed cross-split near duplicates: two validation and four test records, leaving 7,339 images. The reviewed test is a sensitivity subset of an already-inspected test, not a new held-out dataset. Origin 0 was preserved; visual alignment does not definitively settle a one-pixel annotation-origin convention.

Images are accessed through the existing junction `data/raw/ExDark_extracted/ExDark` pointing to the sibling backup. Manifests/caches/legacy runs stay in `selight_project_backup_20261002`. No dataset was duplicated or relabelled for the new comparison. The fresh sample uses the first 200 validation IDs in the project's stable SHA256 ordering. Its record list and intersection with documented near-duplicate omissions are saved in `evidence.json`. Validation already participated in policy selection; these new results are exploratory paired validation comparisons, not a fresh untouched final test.

## Evidence scopes

| Evidence | Number of images | Source / interpretation |
|---|---:|---|
| Full legacy GPU/CPU test | 2,560 | Existing raw predictions and timings; metrics recomputed now. No new test inference or tuning. |
| Full development ablations | 1,792 validation | Cached route predictions; five fitted variants and fixed/threshold baselines. No fresh deployment timing. |
| Reviewed-split test sensitivity | 2,556 | Inherited legacy predictions after documented removals; no new independent test. |
| New five-method comparison | 200 validation | Fresh synchronized GPU inference; three timing repeats, identical predictions across repeats; same detector/config/image IDs. |
| Historical synthetic stress | 30 validation | Native, max-side 640, max-side 320, and 320 + Gaussian sigma5 + JPEG50. Not a calibrated camera model. |
| Demo/video timing | 60 slideshow frames | Independent-frame file processing; playback FPS is not throughput; no real capture-session accuracy. |

Do not combine these rows into one ranking. Different image sets, devices and timing scopes support different claims. Each comparison table identifies its scope.

## Main empirical findings

In the fresh 200-image GPU comparison, AP50:95 is 43.019% for original, 43.016% for gamma, 43.093% for CLAHE, 43.019% for SELight and 42.328% for the Zero-DCE++ adaptation. Mean processing time is approximately 8.55, 9.36, 15.19, 10.92 and 30.23 ms respectively. This timing includes selection/enhancement, transfers, quantization and synchronized detection, and excludes decode, acquisition, startup, metrics, drawing and writing. Repetition means and p95 values are retained; native image sizes strongly affect enhancer cost. Repeats do not create independent accuracy samples.

Paired 95% percentile intervals in `evidence.json` resample the 200 matched images 1,000 times and recompute dataset-level COCO AP. They assume independent images because capture-session labels are unavailable. Individual image F1 bootstrap intervals are separate. The identical selector has an analytically zero AP difference and interval. Interpret all other differences alongside their intervals and the exploratory scope; small point-estimate gains are not proof of an improvement.

Legacy GPU original AP50:95 is approximately 36.98%; fixed CLAHE is approximately 37.35%; selector matches original. The full validation ablation variants all return original. Oracle validation mean-image-F1 headroom is annotation-aware and cannot be claimed as achieved runtime performance or an AP bound. The existing CPU selector mean contains a roughly 69.6-minute decode interruption; it must not be interpreted as ordinary selector computation.

## Why the result may have failed

The dominant route label is original. A shallow classification tree with minimum leaf size and eight global luminance/gradient statistics may not separate objects that benefit from different transforms. The F1 label proxy also differs from the evaluation AP objective, and many images have no target-class boxes. These are plausible explanations to investigate, not causes isolated by this study. Removing cost, removing the gain margin, using brightness alone and removing CLAHE did not change the selected validation outcome.

Cost estimates were obtained on CPU and transferred to GPU. A true cost-aware deployment comparison would retrain on deployment-specific development timing while holding a final session test untouched. Any new feature, utility regression, route balancing or detector fine-tuning study must begin with a declared development protocol; it cannot inherit the already-observed ExDark test as an untouched final confirmation.

## Related methods and comparison status

| Method | Status here | Fair interpretation |
|---|---|---|
| Original COCO YOLOv8n | Measured | Essential reference with no enhancement |
| Gamma correction | Measured | Fixed cheap intensity transform |
| Lab CLAHE | Measured | Fixed local luminance contrast method |
| Brightness threshold / brightness CART / best fixed | Existing validation/stress evidence | Development-selected lightweight controls |
| Zero-DCE++ | Measured | Official architecture and Epoch99 checkpoint, explicitly adapted by replicate padding/output crop |
| SCI | Related work only | No local detection score or speed claim |
| Retinexformer | Related work only | No local detection score or speed claim |
| Larger/fine-tuned detectors | Not measured | Would change detector capacity/training; a separate experiment |

Enhancement papers' published PSNR/SSIM, datasets and FPS cannot be copied into this detection table. Cross-paper comparisons require identical data, class set, detector and timing protocol. Primary references and method provenance are linked in `PAPER.md`.

## Showcase and defensible presentation

The showcase contains matched AP/time plots, paired intervals, legacy class-wise AP, synthetic-stress degradation and annotated case strips. Yellow dashed boxes are GT; cyan boxes are predictions at confidence >= .25. Cases are selected by declared rules: largest Zero-DCE++ F1 gain, largest decline, hardest positive original example and first hash-sampled target-class examples. They are diagnostic cases, not a statistically representative or success-only sample. The case IDs and selection rules are saved in `gallery.json`.

An accurate demonstration statement is: "This project builds and audits a selective-enhancement pipeline for frozen low-light detection. The current tree collapses to original input, and the controlled comparisons reveal quality/cost tradeoffs rather than a proven adaptive advantage." Show the unchanged detector reference and the class mistakes as part of the presentation.

## What remains before submission

Author information and target-venue formatting are not supplied. Genuine phone recordings with session-grouped annotation, an untouched final session test, broader independently reproduced enhancement baselines and a justified algorithmic advance would strengthen the paper. Exposure, resolution, blur, compression, sensor sensitivity and camera price must be distinguished. Resizing/noise stress cannot establish cheap-camera replacement, and no method can be assumed to recover absent signal.

The current manuscript is suitable for faculty review as a prototype/negative-results study. It is not represented as accepted, submission-ready, state-of-the-art or a successful night-vision replacement. Preserve all data/run provenance and keep the production policy frozen until a new development protocol is declared.

## External dataset extension

The frozen model now has human-labelled external transfer runs: 200 original-resolution LLVIP visible test images from the pinned McCheng/LLVIP Hugging Face archive, and all 170 official Penn-Fudan images (423 mask-derived person boxes (345 original plus 78 official small/occluded additions)). Only person is scored. Penn-Fudan is a daylight control. No retraining is performed; no scene-independent generalization or verified absence from detector pretraining is claimed. `external_results.json` and `external_comparison.csv` hold metrics, paired image-conditional intervals and provenance. Sequential `_serial` runs are authoritative; earlier overlapping runs are excluded from published timing/results. `scripts/import_external_datasets.py`, `external_audit.py`, `evaluate_external.py` and `build_external_report.py` reproduce the extension. Exact overlap with the reference ExDark manifest and within-cohort file duplicates is rejected. All 33 software tests pass. See manuscript section 5.6 and RUNBOOK.md.
