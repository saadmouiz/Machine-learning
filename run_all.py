"""Pipeline complet à lancer depuis la racine du projet."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.dataset import download_voc, extract_crops
from src.evaluate import generate_all_figures
from src.predict import format_prediction, load_model, predict_image
from src.train import train_all
from src.config import DEMO_DIR


def main():
    voc_root = download_voc()
    extract_crops(voc_root)
    train_all()
    generate_all_figures()
    model = load_model()
    images = sorted(DEMO_DIR.glob("*.jpg")) + sorted(DEMO_DIR.glob("*.png"))
    print("\n=== Démonstration sur images nouvelles (jeu de test) ===")
    for path in images:
        label, conf, _ = predict_image(path, model=model)
        print(format_prediction(path, label, conf))


if __name__ == "__main__":
    main()
