import os
import zipfile
import glob
import yaml
import random
import shutil
from pathlib import Path
from ultralytics import YOLO

DATASET_ZIP = "dataset.zip"
DATASET_DIR = "dental_dataset"
EPOCHS      = 50
IMG_SIZE    = 640
BATCH       = 16


def extract_dataset():
    if not os.path.exists(DATASET_DIR):
        print("extracting dataset...")
        with zipfile.ZipFile(DATASET_ZIP, "r") as zf:
            zf.extractall(DATASET_DIR)

    # check if train/val split exists already
    train_path = Path(DATASET_DIR) / "images" / "train"
    if not train_path.exists():
        print("no train/val split found, creating 80/20 split...")
        split_dataset()


def split_dataset():
    root       = Path(DATASET_DIR)
    images_dir = root / "images"
    labels_dir = root / "labels"

    for split in ["train", "val"]:
        (images_dir / split).mkdir(exist_ok=True)
        (labels_dir / split).mkdir(exist_ok=True)

    all_images = [f for f in images_dir.glob("*") if f.is_file()]
    random.shuffle(all_images)
    split_idx  = int(len(all_images) * 0.8)

    for img in all_images[:split_idx]:
        shutil.move(str(img), str(images_dir / "train" / img.name))
        lbl = labels_dir / (img.stem + ".txt")
        if lbl.exists():
            shutil.move(str(lbl), str(labels_dir / "train" / lbl.name))

    for img in all_images[split_idx:]:
        shutil.move(str(img), str(images_dir / "val" / img.name))
        lbl = labels_dir / (img.stem + ".txt")
        if lbl.exists():
            shutil.move(str(lbl), str(labels_dir / "val" / lbl.name))

    print(f"split done: {len(all_images[:split_idx])} train, {len(all_images[split_idx:])} val")


def get_data_yaml():
    root = Path(DATASET_DIR).resolve()

    existing = list(root.rglob("data.yaml"))
    if existing:
        return str(existing[0])

    # count classes from label files
    class_ids = set()
    for lf in root.rglob("*.txt"):
        for line in lf.read_text().splitlines():
            parts = line.strip().split()
            if parts:
                try: class_ids.add(int(parts[0]))
                except: pass

    nc    = max(class_ids) + 1 if class_ids else 32
    names = [f"tooth_{i}" for i in range(nc)]

    data = {
        "path":  str(root),
        "train": str(root / "images" / "train"),
        "val":   str(root / "images" / "val"),
        "nc":    nc,
        "names": names,
    }
    yaml_path = str(root / "data.yaml")
    with open(yaml_path, "w") as f:
        yaml.dump(data, f, default_flow_style=False)

    print(f"created data.yaml with {nc} classes")
    return yaml_path


if __name__ == "__main__":
    extract_dataset()
    data_yaml = get_data_yaml()

    model   = YOLO("yolov8m.pt")
    results = model.train(
        data=data_yaml,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH,
        project="rayoscan_tooth",
        name="yolov8m_final",
        patience=15,
        optimizer="AdamW",
        lr0=1e-3,
        lrf=1e-2,
        weight_decay=5e-4,
        mosaic=1.0,
        mixup=0.1,
        copy_paste=0.1,
        fliplr=0.5,
        degrees=10.0,
        hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
        save=True,
        exist_ok=True,
    )

    print("training done.")
    print("best weights: runs/detect/rayoscan_tooth/yolov8m_final/weights/best.pt")