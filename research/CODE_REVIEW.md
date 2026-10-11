# Code and research review

12 October 2026. Review scope: all owned Python modules and tests, configs, exported policy, optional enhancer adapter, existing outputs and preserved experiment evidence. Production routing/training was not tuned in this review.

## Verified behavior

* The existing suite passes **31 tests**. Synthetic/fake-detector tests check software behavior, not pretrained-detector accuracy.
* Gamma uses `y = x**0.65` and brightens uint8 RGB channels with a lookup table. CLAHE modifies only the L channel in Lab using clip 2 and an 8x8 tile grid. Both preserve geometry.
* Feature extraction resizes to 64x64, converts to grayscale, and returns mean, standard deviation, p10/p50/p90, fraction below .2, fraction above .95, and mean finite-difference gradient magnitude.
* Training uses only train/validation cached predictions and explicitly rejects test records and exact train/validation overlap. The original detector is frozen. COCO class IDs are validated and remapped consistently.
* COCO AP is computed by pycocotools over the five-class subset. AP is not estimated from per-image F1. The AP bootstrap recomputes dataset-level ranking/precision after resampling; its test compares that implementation against explicit duplicated-image COCO evaluation.
* The new validation run audits retained content hashes, validates dimensions, checks policy/manifest/checkpoint compatibility, uses identical images/configs across methods, warms every method and synchronizes CUDA around timing.

## Findings that affect the paper

| Finding | Evidence | Implication / next step |
|---|---|---|
| Saved selector has no enhancement behavior | Every exported leaf is `original`; original and selector prediction dictionaries match exactly | Report a negative result. A routing benefit has not been established. |
| Routing adds cost without improving current detections | Fresh comparison times original and selector separately; identical predictions | Do not claim this selector is faster or more accurate. Tree simplification is a possible engineering optimization, not a detection contribution. |
| F1 training proxy differs from AP | Macro F1 and COCO AP rank some methods differently | Declare AP as the main evaluation outcome and retain F1 as the training proxy. Do not silently retune after seeing test results. |
| Target-negative images affect mean-image F1 | Empty GT and empty predictions receive utility 1; 785/1792 validation images have no target-class boxes | Keep negatives for false-positive measurement, but also report micro F1, AP and positive/negative strata. |
| Majority route labels and global features may hide local needs | Original is the dominant label; all leaves collapse despite oracle F1 headroom | This is a hypothesis, not a demonstrated cause. Utility regression/local features require a separate development study and untouched final set. |
| Cost estimates came from CPU | Policy records `profile_device=cpu`; deployed GPU timings differ | GPU transfer of a CPU cost model is explicit. Future cost-aware training should use deployment-device profiles. |
| Original test has already been examined | Historical outputs and review exist | These are preserved test results, not an untouched final confirmation of a new model. New tuning requires new held-out capture sessions. |
| Near duplicates exist beyond exact hashes | Previous visual review identified six cross-split pairs | Report the documented reviewed split as a sensitivity analysis; perceptual search is not exhaustive scene verification. |
| Annotation origin remains uncertain by one pixel | Origin 0 was used; overlays support gross alignment only | Preserve the convention/provenance. Resolve from authoritative source semantics before any new conversion; do not claim definitive origin verification. |
| Severe historical CPU timing interruption | One decode event lasts about 69.6 minutes | Keep raw latency values and label the interruption. Do not interpret that mean as ordinary selector compute cost or silently remove it. |
| Timing scopes differ | Legacy includes decode; fresh/external/stress processing does not | Keep tables separate. File-video timing excludes acquisition/display/drawing/encoding. |
| Optional baseline is an adaptation | Zero-DCE++ replicate-pads instead of official input cropping | Name it `zero_dcepp_scale12_pad`; do not claim exact published-protocol reproduction. |

## Engineering limitations

`fetch_zero_dcepp.py` resolves the current upstream `main` and executes at import time. It records a commit and hashes after fetching, but rerunning it is not guaranteed to reproduce the existing pinned assets. It should eventually expose a `main()` and an explicit commit argument with safe staging. The already-installed assets are hash-verified and were reused without running this fetcher.

`build_cache()` warms fixed transforms before collecting development timings; the legacy `evaluate()` warmup does not separately warm every selector baseline. The new matched benchmark does warm each complete method. These distinct protocols are recorded, and new timings do not replace old timings.

The detector's explicit `half=False` currently produces deprecation messages in the installed Ultralytics API. It still executes FP32 and completed the comparison. Any API migration should be checked against output parity before changing a frozen benchmark implementation.

Video demo output is annotated independent-frame inference. It has no tracking, temporal enhancement, licence-plate recognition, Android execution, energy measurement or calibrated lux data. The supplied clip is an ExDark slideshow. Dataset resizing/noise/JPEG is synthetic stress, not a measured cheap-camera sensor model.

## Submission status

The code supports a reproducible prototype/negative-results study. A publishable improvement, state-of-the-art claim, camera-replacement claim or general-purpose night-vision claim is unsupported. Author identities, target-venue requirements, genuine session-level phone data, and any new algorithm validation remain to be supplied before a final submission.
