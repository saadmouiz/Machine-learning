"""Paramètres globaux du projet."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
CROPS_DIR = DATA_DIR / "processed" / "crops"
DEMO_DIR = DATA_DIR / "demo"
MODELS_DIR = ROOT / "models"
FIGURES_DIR = ROOT / "reports" / "figures"
REPORTS_DIR = ROOT / "reports"

VOC_YEAR = "2007"
VOC_MIRRORS = {
    "trainval": [
        "https://data.brainchip.com/dataset-mirror/voc/VOCtrainval_06-Nov-2007.tar",
        "http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar",
        "https://pjreddie.com/media/files/VOCtrainval_06-Nov-2007.tar",
    ],
}

# Mapping VOC (anglais) -> classes du projet (français)
VOC_TO_PROJECT = {
    "car": "voiture",
    "person": "pieton",
    "bicycle": "cycliste",
    "motorbike": "moto",
    "bus": "bus",
}

CLASSES = ["voiture", "pieton", "cycliste", "moto", "bus"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}

IMAGE_SIZE = (64, 64)
MIN_BBOX_SIZE = 24
MAX_IMAGES_PER_CLASS = 700
RANDOM_STATE = 42
TEST_SIZE = 0.15
VAL_SIZE = 0.15
