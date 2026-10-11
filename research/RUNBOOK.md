# Run and reproduce SELight

Use PowerShell from the project root. The existing environment works; do not recreate it.

```powershell
cd "E:\first paper\02_SELight_Project_Antigravity\selight_project"
.\.venv\Scripts\python.exe -m selight doctor
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider
```

Expected: CUDA available on RTX 4050 and 31 passing tests. Tests establish implementation behavior, not real-world detection accuracy.

## Inspect the results

Open `research/showcase.html` in a browser, `research/PAPER.md` for the manuscript, `research/paper.tex` in the Codex LaTeX editor, and `research/CODE_MAP.md` for the tree and module contracts. Figure PNGs are previewable; companion PDFs are vector plots for later export. `comparison.csv` carries explicit experiment/timing scopes. `evidence.json` binds the input files by SHA256.

The shareable manuscript is `output/pdf/SELight_research_paper.pdf`, exported from Markdown with ReportLab and visually checked. It is independent of the native LaTeX compiler, which did not return a result during this session; TeX compilation remains unverified. The editable `.tex` source is preserved.

## Own photo and video

```powershell
.\.venv\Scripts\python.exe infer_photo.py --image "C:\path\photo.jpg" --config configs/gpu.json --out outputs/my_photo
.\.venv\Scripts\python.exe -m selight demo --source "C:\path\video.mp4" --config configs/gpu.json --out outputs/my_video/annotated.mp4
```

Replace input paths. Use a new photo output folder for every run. Photo results: `annotated.png` and `detections.json`. The JSON collects raw predictions at .001 confidence; displayed boxes use .25. CPU execution uses `configs/cpu.json`. Default policy: `models/policy.json`.

The supplied sample is immediately runnable:

```powershell
.\.venv\Scripts\python.exe infer_photo.py --image "data\raw\ExDark_extracted\ExDark\Bicycle\2015_00375.jpg" --config configs/gpu.json --out outputs/sample_new
```

## Fresh matched comparison

The completed run is `research/experiments/validation_200/`. Use a **new** directory for reproduction:

```powershell
.\.venv\Scripts\python.exe -m scripts.research_compare --manifest "..\selight_project_backup_20261002\data\manifest.json" --policy models/policy.json --config configs/gpu.json --limit 200 --repeats 3 --out research/experiments/validation_200_repeat
```

This compares original, gamma, CLAHE, SELight and the official Zero-DCE++ padding adaptation on the same stable 200 validation IDs. It performs a content audit and never trains or infers on test data. Three repeats produce 3,000 image/method timing records. Method and image order are randomized with seed42; ten warmups per method are excluded. Stage times are synchronized. Read `sample.json`, raw prediction JSONs, `timings.csv` and `summary.json` to verify IDs, routes, actual settings and repetition stability.

The existing detector API prints a deprecation notice for `half=False`; the run still uses FP32. Save its output to a log if desired.

## Rebuild the figures and audit from the completed run

```powershell
.\.venv\Scripts\python.exe -m scripts.build_research_pack --backup "..\selight_project_backup_20261002" --out research --resamples 1000
```

This expects the completed run at `<out>/experiments/validation_200/` and existing full validation outputs under `outputs/ablations/`. It recalculates historical test metrics with the original stable image order, verifies summaries/provenance, computes paired intervals, and regenerates plots/gallery/code inventory. It does not modify production code or archived inputs. It can overwrite its own generated evidence/figure files. Authored Markdown, HTML and LaTeX remain separate presentation files; if input results change, update them explicitly against the new evidence and refresh final artifact hashes.

To refresh the authored manuscript's marked tables, standalone LaTeX plots and the showcase from the completed evidence:

```powershell
.\.venv\Scripts\python.exe -m scripts.render_research_documents --out research
```

This preserves the manuscript prose, replaces its named result tables, redraws inline LaTeX figures and refreshes the code/artifact inventory. Reassess prose claims whenever the evidence changes. The standalone TeX contains inline vector plots so the native compiler needs no additional project files.

The PDF exporter uses the bundled document Python runtime (ReportLab and pypdf), separate from the ML environment:

```powershell
& "C:\Users\chand\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe" -m scripts.export_research_pdf --source research/PAPER.md --out output/pdf/SELight_research_paper.pdf
```

Its default fonts are the existing Windows Times New Roman family. Re-render and inspect pages after changing the manuscript; text extraction alone does not verify layout. `pdf_validation.json` records the Markdown/PDF hashes and review state. The currently delivered PDF has been visually checked.

## Optional existing experiment tools

```powershell
.\.venv\Scripts\python.exe camera_benchmark.py --help
.\.venv\Scripts\python.exe -m scripts.development_ablations --help
.\.venv\Scripts\python.exe -m scripts.external_pilot --help
.\.venv\Scripts\python.exe -m scripts.visual_audit --help
.\.venv\Scripts\python.exe paired_ap_comparison.py --help
.\.venv\Scripts\python.exe video_benchmark.py --help
```

Archived manifest absolute roots resolve through the existing dataset junction. Preserve that junction or restore the original data location; changing manifest bytes changes its hash and invalidates matching experiment provenance. The reviewed manifest requires its matching reviewed policy/cache. Do not disable these checks.

## Verification checklist

External datasets and frozen evaluation can be reproduced from the project root:

```powershell
.\.venv\Scripts\python.exe -B -m scripts.import_external_datasets --limit 200
.\.venv\Scripts\python.exe -B -m scripts.evaluate_external --manifest data/external/llvip_visible_test/manifest.json --reference-manifest ../selight_project_backup_20261002/data/manifest.json --out research/experiments/llvip_visible_200_rerun --repeats 3
.\.venv\Scripts\python.exe -B -m scripts.evaluate_external --manifest data/external/penn_fudan/manifest.json --reference-manifest ../selight_project_backup_20261002/data/manifest.json --out research/experiments/penn_fudan_170_rerun --repeats 3
.\.venv\Scripts\python.exe -B -m scripts.build_external_report --resamples 500
```

Run evaluations sequentially to avoid GPU contention. Output directories must be new. The report builder reads the delivered `llvip_visible_200_serial` and `penn_fudan_170_serial` runs; change its folder mapping explicitly to publish reruns. The generic document renderer rebuilds the main HTML; run the external report builder afterwards to restore its external-results section. Export and visually review the PDF after updating the manuscript. External metrics are person-only. The HF archive is revision-pinned and each selected image has a SHA256; Penn-Fudan archive SHA256 is recorded in its manifest. Imported data are research inputs, not production training changes.

1. Compare policy/checkpoint/manifest hashes and actual environment, not filenames alone.
2. Keep AP and mean-image F1 separate; inspect positive and negative strata.
3. Compare identical image IDs and detector settings.
4. Preserve score-tie/image ordering when reproducing saved COCO AP.
5. Keep processing-only and decode-inclusive timing tables separate.
6. Treat timing repetitions as repeated cost measurements, not independent accuracy samples.
7. Interpret bootstrap intervals as image-conditional; session correlation is unverified.
8. Keep new tuning away from the already-inspected ExDark test; collect untouched phone sessions for final confirmation.
