# Dental Tooth Numbering Detection — RayoScan AI

YOLOv8m trained on 497 dental X-ray images for automated tooth detection using FDI numbering.

**mAP@0.5: 0.8134 | mAP@0.5:0.95: 0.5333**

## Live Demo
https://discomenon-rayoscan-tooth-detection.hf.space

## Model Weights
[[Download best.pt](https://drive.google.com/file/d/1akqrQU1jicGFIOjX5L_BkrIdBNKt04gJ/view?usp=sharing)]

## Usage
pip install -r requirements.txt
python train.py
python inference.py --weights best.pt --source path/to/images/

## Report
See RayoScan_Report.pdf for full methodology and results.
