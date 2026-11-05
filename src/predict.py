"""Prédiction exploitable : Image → Classe (confiance %)."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np

from src.config import CLASSES, DEMO_DIR, MODELS_DIR
from src.features import extract_one


def load_model():
    path = MODELS_DIR / "best_model.joblib"
    if not path.exists():
        raise FileNotFoundError("Modèle introuvable. Lancez d'abord : python -m src.train")
    return joblib.load(path)


def predict_image(image_path: str | Path, model=None) -> tuple[str, float, dict]:
    model = model or load_model()
    feats = extract_one(image_path).reshape(1, -1)
    pred_idx = int(model.predict(feats)[0])
    label = CLASSES[pred_idx]
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(feats)[0]
    else:
        proba = np.zeros(len(CLASSES))
        proba[pred_idx] = 1.0
    scores = {CLASSES[i]: float(p) for i, p in enumerate(proba)}
    return label, float(proba[pred_idx]), scores


def format_prediction(image_path, label: str, confidence: float) -> str:
    return f"{Path(image_path).name} → {label.capitalize()} ({confidence * 100:.0f} %)"


def main():
    parser = argparse.ArgumentParser(description="Classifier un objet routier")
    parser.add_argument("images", nargs="*", help="Chemins d'images à classifier")
    parser.add_argument("--demo", action="store_true", help="Prédire les images du dossier data/demo")
    args = parser.parse_args()

    paths = [Path(p) for p in args.images]
    if args.demo or not paths:
        paths = sorted(DEMO_DIR.glob("*.jpg")) + sorted(DEMO_DIR.glob("*.png"))
    if not paths:
        raise SystemExit("Aucune image fournie. Utilisez --demo ou passez des chemins.")

    model = load_model()
    for path in paths:
        label, conf, scores = predict_image(path, model=model)
        print(format_prediction(path, label, conf))
        details = ", ".join(f"{k}={v * 100:.1f}%" for k, v in scores.items())
        print(f"    scores : {details}")


if __name__ == "__main__":
    main()
