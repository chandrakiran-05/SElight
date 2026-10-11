# Algorithm and dependency attribution

Gamma correction is an established power-law image transform. Implementation
reference: https://docs.opencv.org/4.x/d3/dc1/tutorial_basic_linear_transform.html

CLAHE: Karel Zuiderveld, Graphics Gems IV, 1994, pages 474–485.
https://doi.org/10.1016/B978-0-12-336156-1.50061-6
OpenCV implementation: https://docs.opencv.org/4.x/d5/daf/tutorial_py_histogram_equalization.html

DecisionTreeClassifier: https://scikit-learn.org/stable/modules/tree.html
YOLOv8 and pretrained weights: https://docs.ultralytics.com/models/yolov8/
Ultralytics licensing information: https://www.ultralytics.com/license
COCO evaluation: https://github.com/cocodataset/cocoapi
ExDark: https://github.com/cs-chan/Exclusively-Dark-Image-Dataset

Project-specific glue code was written for this project. Downloaded external
research assets are explicitly listed below and were not published or packaged
for redistribution by this task. Respect each model and dataset's own terms. Cite the
dataset paper and the algorithm sources in academic submissions. Do not describe
this package as an official reproduction of IA-YOLO, ERUP-YOLO or any other paper.

Local review additions (2 October 2026): `third_party/zero_dcepp/model.py` and
`Epoch99.pth` are the official Zero-DCE++ research assets, pinned at commit
09f202b690f82da939b8e6ec8535960ae97ad8bd from
https://github.com/Li-Chongyi/Zero-DCE_extension . The official README states
academic research use under CC BY-NC 4.0. Source URLs and hashes are in its
provenance.json. Cite Li, Guo and Loy, Learning to Enhance Low-Light Image via
Zero-Reference Deep Curve Estimation, DOI 10.1109/TPAMI.2021.3063604.
The adapter changes input cropping to replicated padding/output cropping and is
explicitly named `zero_dcepp_scale12_pad`; it is not an exact official test
reproduction. These downloaded assets stay local and are not published by this task.
