from ultralytics import YOLO
from roboflow import Roboflow
from pathlib import Path
from dotenv import load_dotenv
import os
import shutil

load_dotenv()

ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY")
WORKSPACE = "seun-filaoye-pghzu"
PROJECT = "chip-detection-5ykcx"
VERSION = 8

PROJECT_ROOT = Path.home() / "Vision" / "chip_detection"
DATASETS_DIR = PROJECT_ROOT / "datasets"
MODELS_DIR = PROJECT_ROOT / "models"

DATASETS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

if not ROBOFLOW_API_KEY:
    raise ValueError("ROBOFLOW_API_KEY not found in .env file")

print("loading Roboflow workspace...")
rf = Roboflow(api_key=ROBOFLOW_API_KEY)

print("loading Roboflow project...")
project = rf.workspace(WORKSPACE).project(PROJECT)
version = project.version(VERSION)

dataset_folder = DATASETS_DIR / f"{PROJECT}-{VERSION}"

if dataset_folder.exists():
    print(f"Removing old dataset folder: {dataset_folder}")
    shutil.rmtree(dataset_folder)

print(f"Downloading dataset to: {dataset_folder}")
dataset = version.download("yolov11", location=str(dataset_folder))

dataset_path = Path(dataset.location)
print(f"Dataset downloaded to: {dataset_path}")

data_yaml = dataset_path / "data.yaml"
if not data_yaml.exists():
    raise FileNotFoundError(f"Could not find data.yaml at {data_yaml}")

train_dir = dataset_path / "train" / "images"
valid_dir = dataset_path / "valid" / "images"
test_dir = dataset_path / "test" / "images"

print("Checking dataset folders...")
print(f"train exists: {train_dir.exists()} -> {train_dir}")
print(f"valid exists: {valid_dir.exists()} -> {valid_dir}")
print(f"test exists:  {test_dir.exists()} -> {test_dir}")

if not train_dir.exists() or not valid_dir.exists():
    raise FileNotFoundError(
        "Dataset download is incomplete. Missing train/images or valid/images folders."
    )

yaml_text = data_yaml.read_text()
yaml_text = yaml_text.replace("train: ../train/images", f"train: {train_dir}")
yaml_text = yaml_text.replace("val: ../valid/images", f"val: {valid_dir}")
yaml_text = yaml_text.replace("test: ../test/images", f"test: {test_dir}")
data_yaml.write_text(yaml_text)

print(f"Updated data.yaml: {data_yaml}")

training_args = {
    "data": str(data_yaml),
    "epochs": 300,
    "imgsz": 640,
    "batch": 16,
    "lr0": 0.01,
    "lrf": 0.01,
    "momentum": 0.937,
    "weight_decay": 0.0005,
    "warmup_epochs": 3,
    "box": 7.5,
    "cls": 1.5,
    "dfl": 1.5,
    "patience": 30,
    "save": True,
    "save_period": 10,
    "cache": False,
    "workers": 4,
    "exist_ok": True,
    "pretrained": True,
    "optimizer": "AdamW",
    "verbose": True,
    "seed": 0,
    "deterministic": False,
    "single_cls": False,
    "rect": False,
    "cos_lr": False,
    "close_mosaic": 10,
    "resume": False,
    "amp": True,
    "fraction": 1.0,
    "profile": False,
    "freeze": 0,
    "multi_scale": False,
    "val": True,
    "hsv_h": 0.015,
    "hsv_s": 0.4,
    "hsv_v": 0.4,
    "degrees": 5,
    "translate": 0.05,
    "scale": 0.2,
    "shear": 2,
    "perspective": 0.0,
    "flipud": 0.0,
    "fliplr": 0.5,
    "mosaic": 0.2,
    "mixup": 0.0,
    "copy_paste": 0.0,
    "erasing": 0.2,
}

model = YOLO("yolo11s.pt")
results = model.train(**training_args)

best_src = Path(model.trainer.best)
best_dst = MODELS_DIR / "best.pt"

if best_src.exists():
    shutil.copy(best_src, best_dst)
    print(f"Best model copied to: {best_dst}")
else:
    print("Warning: best.pt not found.")
