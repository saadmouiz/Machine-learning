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


# Nombre minimal de fichiers attendu dans l'archive VOC.
# On vérifie qu'on a bien au moins plusieurs milliers d'images et d'annotations,
# ce qui permet d'éviter d'utiliser une archive incomplète ou tronquée.
_MIN_VOC_FILES = 4000


def _is_complete_voc_archive(archive: Path) -> bool:
    """Vérifie qu'une archive VOC est complète et exploitable.

    L'archive Pascal VOC contient, pour le jeu d'entraînement, environ 5 000 images
    et 5 000 fichiers XML d'annotations. Si l'archive est partielle ou corrompue,
    on ne doit pas la considérer comme valide.
    """
    try:
        with tarfile.open(archive, "r:*") as tar:
            # On liste tous les fichiers contenus dans l'archive et on normalise les chemins.
            names = [
                member.name.replace("\\", "/").lower()
                for member in tar.getmembers()
                if member.isfile()
            ]
        # On compte le nombre d'images JPEG et le nombre d'annotations XML.
        image_count = sum("/jpegimages/" in f"/{name}" and name.endswith(".jpg") for name in names)
        annotation_count = sum("/annotations/" in f"/{name}" and name.endswith(".xml") for name in names)
        return image_count >= _MIN_VOC_FILES and annotation_count >= _MIN_VOC_FILES
    except (OSError, tarfile.TarError):
        # L'ouverture du tar peut échouer si le fichier est cassé, incomplet ou non-zippe.
        return False


def _download(url: str, dest: Path) -> Path:
    # On s'assure que le dossier de destination existe pour stocker l'archive VOC.
    dest.parent.mkdir(parents=True, exist_ok=True)

    # Si l'archive finale est déjà présente et complète, on ne la télécharge pas à nouveau.
    if dest.exists() and _is_complete_voc_archive(dest):
        print(f"[OK] Déjà téléchargé : {dest.name}")
        return dest

    # Si le fichier existe mais est incomplet, on le supprime avant de recommencer le téléchargement.
    if dest.exists():
        dest.unlink()

    # On travaille d'abord sur une archive intermédiaire ".part" ; cela permet de reprendre
    # un téléchargement interrompu sans repartir de zéro.
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
        # HTTP 206 = reprise d'un téléchargement partiel. Si ce code n'est pas renvoyé,
        # on doit recommencer depuis le début pour éviter un fichier invalide.
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

    # On vérifie que l'archive finale est bien complète avant de la considérer comme valide.
    if not _is_complete_voc_archive(partial):
        raise tarfile.ReadError("Archive VOC incomplète ou invalide ; le fichier partiel est conservé")
    partial.replace(dest)
    return dest


def _extract_tar(archive: Path, out_dir: Path) -> Path:
    # Emplacement de la structure standard de Pascal VOC après extraction.
    voc_root = out_dir / "VOCdevkit" / "VOC2007"
    stamp = archive.with_name(archive.name + ".extracted")
    image_dir = voc_root / "JPEGImages"
    annotation_dir = voc_root / "Annotations"

    # Si l'archive a déjà été extraite correctement, on évite une extraction inutile.
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
            # Python 3.12+ prend en charge `filter="data"` pour sécuriser l'extraction.
            tar.extractall(out_dir, filter="data")
        except TypeError:
            # Anciennes versions de Python n'acceptent pas ce paramètre : on retombe sur la version classique.
            tar.extractall(out_dir)

    # Contrôle post-extraction : on s'assure que les images et les annotations sont bien présentes.
    if (
        len(list((voc_root / "JPEGImages").glob("*.jpg"))) < _MIN_VOC_FILES
        or len(list((voc_root / "Annotations").glob("*.xml"))) < _MIN_VOC_FILES
    ):
        raise RuntimeError(f"Extraction VOC incomplète : {archive}")

    stamp.write_text("ok", encoding="utf-8")
    return voc_root


def _download_with_mirrors(urls: list[str], dest: Path) -> Path:
    # On essaie plusieurs URLs (miroirs) car certains peuvent être bloqués ou indisponibles.
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
    """Télécharge et extrait Pascal VOC 2007 trainval.

    On boucle sur la liste des miroirs définis dans la configuration et on garde
    le premier miroir qui fonctionne. La fonction retourne le dossier
    ``VOCdevkit/VOC2007`` contenant les images et les annotations.
    """
    voc_root = None
    for urls in VOC_MIRRORS.values():
        archive = RAW_DIR / Path(urls[0]).name
        _download_with_mirrors(urls, archive)
        voc_root = _extract_tar(archive, RAW_DIR)
    return voc_root


def _iter_objects(annotation_path: Path):
    # Chaque fichier XML Pascal VOC décrit une image avec plusieurs objets.
    # On parse le XML, puis on extrait les objets utiles pour notre projet.
    tree = ET.parse(annotation_path)
    root = tree.getroot()
    filename = root.findtext("filename")

    for obj in root.findall("object"):
        name = obj.findtext("name")
        # On ne garde que les objets qui correspondent à nos classes du projet.
        if name not in VOC_TO_PROJECT:
            continue

        # Les objets marqués "difficult" ne sont pas utiles pour l'entraînement,
        # car ils sont trop complexes ou peu fiables.
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

        # On convertit les objets détectés en un format exploitable :
        # (nom du fichier image, classe projet, boîtes englobantes)
        yield filename, VOC_TO_PROJECT[name], (xmin, ymin, xmax, ymax)


def analyze_image_directory(directory: Path) -> dict:
    """Retourne le nombre total d'images, les images lisibles et celles qui sont corrompues."""
    if not directory.exists():
        raise FileNotFoundError(f"Répertoire introuvable : {directory}")

    # On ne garde que les fichiers image connus : JPG, JPEG, PNG, BMP, WEBP.
    image_paths = sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    )
    readable_paths = []
    unreadable_paths = []

    # On ouvre chaque image avec PIL et on vérifie qu'elle est bien décodable.
    # Si le fichier est corrompu, une exception est levée et on l'ajoute à la liste des illisibles.
    for image_path in image_paths:
        try:
            with Image.open(image_path) as img:
                img.verify()
            readable_paths.append(image_path)
        except Exception:
            unreadable_paths.append(image_path)

    return {
        "directory": directory,
        "total_images": len(image_paths),
        "readable_images": len(readable_paths),
        "unreadable_images": unreadable_paths,
    }


def extract_crops(voc_root: Path | None = None) -> Path:
    """Découpe les boîtes englobantes et enregistre des images 64x64 par classe.

    Cette étape est essentielle pour construire un dataset exploitable en ML :
    - on part des annotations VOC ;
    - on extrait chaque objet détecté ;
    - on le redimensionne à une taille uniforme ;
    - on le range dans un dossier spécifique à sa classe.
    """
    if voc_root is None:
        voc_root = RAW_DIR / "VOCdevkit" / "VOC2007"

    jpeg_dir = voc_root / "JPEGImages"
    ann_dir = voc_root / "Annotations"
    if not jpeg_dir.exists():
        raise FileNotFoundError(
            "Pascal VOC introuvable. Lancez d'abord : python -m src.dataset"
        )

    # Avant de produire les crops, on vérifie si certaines images du dataset sont corroptes.
    raw_stats = analyze_image_directory(jpeg_dir)
    print(f"Lecture du dataset VOC : {raw_stats['total_images']} images détectées")
    print(f"  - images lisibles : {raw_stats['readable_images']}")
    print(f"  - images illisibles : {len(raw_stats['unreadable_images'])}")
    if raw_stats["unreadable_images"]:
        for bad_path in raw_stats["unreadable_images"][:10]:
            print(f"    * {bad_path.name}")
        if len(raw_stats["unreadable_images"]) > 10:
            print(f"    ... et {len(raw_stats['unreadable_images']) - 10} autres fichiers")

    # On prépare le dossier final contenant les images découpées par classe.
    for cls in CLASSES:
        (CROPS_DIR / cls).mkdir(parents=True, exist_ok=True)

    # On nettoie les anciens crops pour repartir proprement à chaque exécution.
    for cls in CLASSES:
        for crop_path in (CROPS_DIR / cls).glob("*.jpg"):
            crop_path.unlink()

    counts = Counter()
    saved = 0
    xml_files = sorted(ann_dir.glob("*.xml"))

    # On parcourt chaque fichier XML d'annotation et on extrait les objets détectés.
    for xml_path in tqdm(xml_files, desc="Extraction des objets"):
        for filename, cls, (xmin, ymin, xmax, ymax) in _iter_objects(xml_path):
            # On s'arrête quand on a atteint le quota maximum pour cette classe.
            if counts[cls] >= MAX_IMAGES_PER_CLASS:
                continue

            # Les boîtes trop petites ne sont pas pertinentes pour l'apprentissage.
            w, h = xmax - xmin, ymax - ymin
            if w < MIN_BBOX_SIZE or h < MIN_BBOX_SIZE:
                continue

            image_path = jpeg_dir / filename
            if not image_path.exists():
                continue

            try:
                with Image.open(image_path) as img:
                    # Les coordonnées VOC utilisent des indices 1-based (inclusifs),
                    # tandis que PIL travaille en coordonnées 0-based avec borne exclusive.
                    # On corrige donc les bornes avant de découper.
                    left = max(0, xmin - 1)
                    top = max(0, ymin - 1)
                    right = min(img.width, xmax)
                    bottom = min(img.height, ymax)

                    if right - left < MIN_BBOX_SIZE or bottom - top < MIN_BBOX_SIZE:
                        continue

                    # Découpe de l'objet dans l'image d'origine, puis redimensionnement à 64x64.
                    crop = img.convert("RGB").crop((left, top, right, bottom))
                    crop = crop.resize(IMAGE_SIZE, Image.Resampling.BILINEAR)
            except Exception:
                print(f"[WARN] Image illisible ignorée : {image_path}")
                continue

            # Nommage explicite de chaque crop pour garder la trace de sa classe et de son numéro.
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
    """Retourne (chemins, labels) à partir des crops extraits.

    Chaque image de crop correspond à une instance d'une classe précise. On reconstruit
    une liste de chemins d'images ainsi qu'une liste des étiquettes correspondantes pour
    alimenter ensuite les modèles de machine learning.
    """
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
    # Point d'entrée principal du module.
    # On télécharge le dataset VOC puis on extrait les objets routiers triés par classe.
    voc_root = download_voc()
    extract_crops(voc_root)


if __name__ == "__main__":
    main()
