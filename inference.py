import os
import random
from pathlib import Path

import cv2
from ultralytics import YOLO

WEIGHTS   = "runs/detect/rayoscan_tooth/yolov8m_final/weights/best.pt"
SOURCE    = "dental_dataset"
OUT_DIR   = "sample_outputs"
CONF      = 0.25
IOU       = 0.45
IMG_SIZE  = 640
N_SAMPLES = 10


def get_images(folder, n):
    exts = {".jpg", ".jpeg", ".png"}
    imgs = [p for p in Path(folder).rglob("*") if p.suffix.lower() in exts]
    random.shuffle(imgs)
    return imgs[:n]


def annotate(img_path, result, out_dir):
    img   = cv2.imread(img_path)
    names = result.names

    for box in result.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        conf  = float(box.conf[0])
        cls   = int(box.cls[0])
        label = f"{names[cls]} {conf:.2f}"

        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 200, 100), 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (x1, y1 - th - 6), (x1 + tw + 4, y1), (0, 200, 100), -1)
        cv2.putText(img, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    out_path = os.path.join(out_dir, Path(img_path).name)
    cv2.imwrite(out_path, img)
    return out_path


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    model  = YOLO(WEIGHTS)
    images = get_images(SOURCE, N_SAMPLES)

    for img_path in images:
        results = model.predict(str(img_path), conf=CONF, iou=IOU,
                                imgsz=IMG_SIZE, verbose=False)
        result  = results[0]
        out     = annotate(str(img_path), result, OUT_DIR)
        print(f"{img_path.name} -> {len(result.boxes)} detections -> {out}")

    print(f"\ndone. {len(images)} images saved to {OUT_DIR}/")