"""Extraction de caractéristiques visuelles (HOG + couleur + texture)."""

from __future__ import annotations

import numpy as np
from PIL import Image
from skimage.feature import hog, local_binary_pattern
from tqdm import tqdm

from src.config import IMAGE_SIZE


def load_rgb(path, size=IMAGE_SIZE) -> np.ndarray:
    with Image.open(path) as img:
        img = img.convert("RGB").resize(size, Image.Resampling.BILINEAR)
        return np.asarray(img, dtype=np.float32) / 255.0


def color_histogram(image: np.ndarray, bins: int = 16) -> np.ndarray:
    hist = []
    for channel in range(3):
        h, _ = np.histogram(image[:, :, channel], bins=bins, range=(0.0, 1.0), density=True)
        hist.append(h)
    return np.concatenate(hist).astype(np.float32)


def hog_features(image: np.ndarray) -> np.ndarray:
    gray = 0.299 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 2]
    feats = hog(
        gray,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        feature_vector=True,
    )
    return feats.astype(np.float32)


def lbp_histogram(image: np.ndarray, n_points: int = 8, radius: int = 1, bins: int = 10) -> np.ndarray:
    gray = 0.299 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 2]
    lbp = local_binary_pattern(gray, n_points, radius, method="uniform")
    n_bins = n_points + 2
    hist, _ = np.histogram(lbp, bins=n_bins, range=(0, n_bins), density=True)
    return hist.astype(np.float32)


def extract_one(path) -> np.ndarray:
    image = load_rgb(path)
    return np.concatenate([hog_features(image), color_histogram(image), lbp_histogram(image)])


def extract_many(paths) -> np.ndarray:
    vectors = [extract_one(p) for p in tqdm(paths, desc="Extraction des descripteurs")]
    return np.vstack(vectors)
