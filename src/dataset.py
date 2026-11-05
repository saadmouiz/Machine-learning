"""Acquisition Pascal VOC 2007 et extraction des objets routiers."""

from __future__ import annotations

import tarfile
import urllib.request
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

from src.config import (
    CLASSES,
    CROPS_DIR,
    IMAGE_SIZE,
    MAX_IMAGES_PER_CLASS,
    MIN_BBOX_SIZE,
    RAW_DIR,
    VOC_MIRRORS,
    VOC_TO_PROJECT,
)


_MIN_VOC_FILES = 4000


def _is_complete_voc_archive(archive: Path) -> bool:
    """Reject truncated archives; VOC 2007 trainval contains about 5,000 of each file."""
    try:
        with tarfile.open(archive, "r:*") as tar:
            names = [
                member.name.replace("\\", "/").lower()
                for member in tar.getmembers()
                if member.isfile()
            ]
        image_count = sum("/jpegimages/" in f"/{name}" and name.endswith(".jpg") for name in names)
        annotation_count = sum("/annotations/" in f"/{name}" and name.endswith(".xml") for name in names)
        return image_count >= _MIN_VOC_FILES and annotation_count >= _MIN_VOC_FILES
    except (OSError, tarfile.TarError):
        return False


def _download(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and _is_complete_voc_archive(dest):
        print(f"[OK] Déjà téléchargé : {dest.name}")
        return dest
    if dest.exists():
        dest.unlink()

    partial = dest.with_suffix(dest.suffix + ".part")
    if partial.exists() and _is_complete_voc_archive(partial):
        partial.replace(dest)
        print(f"[OK] Archive téléchargée : {dest.name}")
        return dest

    offset = partial.stat().st_size if partial.exists() else 0
    headers = {"User-Agent": "Mozilla/5.0 (compatible; PACTE-ML/1.0)"}
    if offset:
        headers["Range"] = f"bytes={offset}-"
        print(f"Reprise du téléchargement à {offset / 1_000_000:.1f} Mo : {url}")
    else:
        print(f"Téléchargement de {url}")
    req = urllib.request.Request(url, headers=headers)

    class _Progress(tqdm):
        def update_to(self, n: int, total: int | None):
            if total:
                self.total = total
            self.update(n - self.n)

    with urllib.request.urlopen(req, timeout=120) as resp:
        append = offset > 0 and resp.status == 206
        if offset and not append:
            print("Le miroir ne permet pas la reprise ; redémarrage du téléchargement.")
        mode = "ab" if append else "wb"
        total_header = resp.headers.get("Content-Length")
        total_n = int(total_header) + (offset if append and total_header else 0) if total_header else None
        content_range = resp.headers.get("Content-Range")
        if content_range and "/" in content_range:
            total_n = int(content_range.rsplit("/", 1)[1])
        with partial.open(mode) as out:
            with _Progress(
                unit="B", unit_scale=True, miniters=1, desc=dest.name,
                total=total_n, initial=offset if append else 0,
            ) as bar:
                while True:
                    chunk = resp.read(1024 * 256)
                    if not chunk:
                        break
                    out.write(chunk)
                    bar.update(len(chunk))
    if not _is_complete_voc_archive(partial):
        raise tarfile.ReadError("Archive VOC incomplète ou invalide ; le fichier partiel est conservé")
    partial.replace(dest)
    return dest


def _extract_tar(archive: Path, out_dir: Path) -> Path:
    voc_root = out_dir / "VOCdevkit" / "VOC2007"
    stamp = archive.with_name(archive.name + ".extracted")
    image_dir = voc_root / "JPEGImages"
    annotation_dir = voc_root / "Annotations"
    if (
        stamp.exists()
        and image_dir.exists()
        and annotation_dir.exists()
        and len(list(image_dir.glob("*.jpg"))) >= _MIN_VOC_FILES
        and len(list(annotation_dir.glob("*.xml"))) >= _MIN_VOC_FILES
    ):
        print(f"[OK] Archive déjà extraite : {archive.name}")
        return voc_root
    print(f"Extraction de {archive.name} ...")
    with tarfile.open(archive, "r") as tar:
        try:
            tar.extractall(out_dir, filter="data")
        except TypeError:
            tar.extractall(out_dir)
    if (
        len(list((voc_root / "JPEGImages").glob("*.jpg"))) < _MIN_VOC_FILES
        or len(list((voc_root / "Annotations").glob("*.xml"))) < _MIN_VOC_FILES
    ):
        raise RuntimeError(f"Extraction VOC incomplète : {archive}")
    stamp.write_text("ok", encoding="utf-8")
    return voc_root


def _download_with_mirrors(urls: list[str], dest: Path) -> Path:
    last_error = None
    for url in urls:
        try:
            return _download(url, dest)
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            print(f"[WARN] Échec du miroir {url} : {exc}")
            if dest.exists():
                dest.unlink()
    raise RuntimeError(f"Impossible de télécharger {dest.name}") from last_error


def download_voc() -> Path:
    """Télécharge et extrait Pascal VOC 2007 trainval."""
    voc_root = None
    for urls in VOC_MIRRORS.values():
        archive = RAW_DIR / Path(urls[0]).name
        _download_with_mirrors(urls, archive)
        voc_root = _extract_tar(archive, RAW_DIR)
    return voc_root


def _iter_objects(annotation_path: Path):
    tree = ET.parse(annotation_path)
    root = tree.getroot()
    filename = root.findtext("filename")
    for obj in root.findall("object"):
        name = obj.findtext("name")
        if name not in VOC_TO_PROJECT:
            continue
        difficult = obj.findtext("difficult")
        if difficult == "1":
            continue
        bbox = obj.find("bndbox")
        if bbox is None:
            continue
        coordinates = [bbox.findtext(key) for key in ("xmin", "ymin", "xmax", "ymax")]
        if any(value is None for value in coordinates):
            continue
        try:
            xmin, ymin, xmax, ymax = (int(float(value)) for value in coordinates)
        except ValueError:
            continue
        yield filename, VOC_TO_PROJECT[name], (xmin, ymin, xmax, ymax)


def extract_crops(voc_root: Path | None = None) -> Path:
    """Découpe les boîtes englobantes et enregistre des images 64x64 par classe."""
    if voc_root is None:
        voc_root = RAW_DIR / "VOCdevkit" / "VOC2007"
    jpeg_dir = voc_root / "JPEGImages"
    ann_dir = voc_root / "Annotations"
    if not jpeg_dir.exists():
        raise FileNotFoundError(
            "Pascal VOC introuvable. Lancez d'abord : python -m src.dataset"
        )

    for cls in CLASSES:
        (CROPS_DIR / cls).mkdir(parents=True, exist_ok=True)

    for cls in CLASSES:
        for crop_path in (CROPS_DIR / cls).glob("*.jpg"):
            crop_path.unlink()

    counts = Counter()
    saved = 0
    xml_files = sorted(ann_dir.glob("*.xml"))
    for xml_path in tqdm(xml_files, desc="Extraction des objets"):
        for filename, cls, (xmin, ymin, xmax, ymax) in _iter_objects(xml_path):
            if counts[cls] >= MAX_IMAGES_PER_CLASS:
                continue
            w, h = xmax - xmin, ymax - ymin
            if w < MIN_BBOX_SIZE or h < MIN_BBOX_SIZE:
                continue
            image_path = jpeg_dir / filename
            if not image_path.exists():
                continue
            with Image.open(image_path) as img:
                # VOC coordinates are 1-based and inclusive; PIL uses 0-based, exclusive bounds.
                left = max(0, xmin - 1)
                top = max(0, ymin - 1)
                right = min(img.width, xmax)
                bottom = min(img.height, ymax)
                if right - left < MIN_BBOX_SIZE or bottom - top < MIN_BBOX_SIZE:
                    continue
                crop = img.convert("RGB").crop((left, top, right, bottom))
                crop = crop.resize(IMAGE_SIZE, Image.Resampling.BILINEAR)
            out = CROPS_DIR / cls / f"{xml_path.stem}__{cls}_{counts[cls]:04d}.jpg"
            crop.save(out, quality=92)
            counts[cls] += 1
            saved += 1

    print("Images extraites par classe :")
    for cls in CLASSES:
        print(f"  - {cls:10s} : {counts[cls]}")
    print(f"Total : {saved} images -> {CROPS_DIR}")
    return CROPS_DIR


def load_image_paths():
    """Retourne (chemins, labels) à partir des crops extraits."""
    paths, labels = [], []
    for cls in CLASSES:
        files = sorted((CROPS_DIR / cls).glob("*.jpg"))
        paths.extend(files)
        labels.extend([cls] * len(files))
    if not paths:
        raise FileNotFoundError(
            f"Aucune image dans {CROPS_DIR}. Exécutez python -m src.dataset"
        )
    return np.array(paths, dtype=object), np.array(labels)


def main():
    voc_root = download_voc()
    extract_crops(voc_root)


if __name__ == "__main__":
    main()
