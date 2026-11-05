"""Graphiques d'évaluation et préparation des images de démonstration."""

from __future__ import annotations

import json
import shutil
from collections import Counter
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image

from src.config import CLASSES, CROPS_DIR, DEMO_DIR, FIGURES_DIR, MODELS_DIR
from src.dataset import load_image_paths

sns.set_theme(style="whitegrid", font_scale=1.05)


def class_counts() -> pd.DataFrame:
    paths, labels = load_image_paths()
    counts = Counter(labels)
    df = pd.DataFrame({"classe": CLASSES, "nombre": [counts[c] for c in CLASSES]})
    df["proportion"] = df["nombre"] / df["nombre"].sum()
    return df


def plot_class_distribution(df: pd.DataFrame | None = None) -> Path:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    df = df if df is not None else class_counts()
    if "proportion" not in df.columns:
        df = df.copy()
        df["proportion"] = df["nombre"] / df["nombre"].sum()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    sns.barplot(data=df, x="classe", y="nombre", hue="classe", ax=axes[0], legend=False, palette="viridis")
    axes[0].set_title("Nombre d'images par classe")
    axes[0].set_xlabel("Classe")
    axes[0].set_ylabel("Nombre d'images")
    axes[1].pie(df["nombre"], labels=df["classe"], autopct="%1.1f%%", startangle=90)
    axes[1].set_title("Répartition des classes")
    fig.tight_layout()
    out = FIGURES_DIR / "repartition_classes.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_examples() -> Path:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(len(CLASSES), 5, figsize=(10, 10))
    for i, cls in enumerate(CLASSES):
        files = sorted((CROPS_DIR / cls).glob("*.jpg"))[:5]
        for j in range(5):
            ax = axes[i, j]
            ax.axis("off")
            if j < len(files):
                ax.imshow(Image.open(files[j]))
            if j == 0:
                ax.set_ylabel(cls, fontsize=11)
    fig.suptitle("Exemples d'objets routiers extraits (Pascal VOC 2007)", y=0.92)
    fig.tight_layout()
    out = FIGURES_DIR / "exemples_images.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_confusion(cm, title: str, filename: str) -> Path:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=CLASSES,
        yticklabels=CLASSES,
        ax=ax,
    )
    ax.set_xlabel("Classe prédite")
    ax.set_ylabel("Classe réelle")
    ax.set_title(title)
    fig.tight_layout()
    out = FIGURES_DIR / filename
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_model_comparison(metrics: dict) -> Path:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, block in metrics["results"].items():
        rows.append(
            {
                "modèle": name,
                "Accuracy": block["test"]["accuracy"],
                "Precision": block["test"]["precision_macro"],
                "Recall": block["test"]["recall_macro"],
                "F1-score": block["test"]["f1_macro"],
            }
        )
    df = pd.DataFrame(rows)
    melted = df.melt(id_vars="modèle", var_name="métrique", value_name="score")
    fig, ax = plt.subplots(figsize=(9, 4.8))
    sns.barplot(data=melted, x="métrique", y="score", hue="modèle", ax=ax)
    ax.set_ylim(0, 1)
    ax.set_title("Comparaison des modèles sur le jeu de test")
    fig.tight_layout()
    out = FIGURES_DIR / "comparaison_modeles.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    df.to_csv(FIGURES_DIR / "comparaison_modeles.csv", index=False)
    return out


def export_demo_images(n_per_class: int = 2) -> list[Path]:
    DEMO_DIR.mkdir(parents=True, exist_ok=True)
    split_paths = joblib.load(MODELS_DIR / "split_paths.joblib")
    test_paths = [Path(p) for p in split_paths["test"]]
    encoder_labels = [p.parent.name for p in test_paths]
    copied = []
    for cls in CLASSES:
        selected = [p for p, lab in zip(test_paths, encoder_labels) if lab == cls][:n_per_class]
        for i, src in enumerate(selected, start=1):
            dest = DEMO_DIR / f"{cls}_{i}{src.suffix}"
            shutil.copy2(src, dest)
            copied.append(dest)
    return copied


def generate_all_figures():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    df = class_counts()
    df.to_csv(FIGURES_DIR / "description_dataset.csv", index=False)
    plot_class_distribution(df)
    plot_examples()
    metrics_path = MODELS_DIR / "metrics.json"
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        plot_model_comparison(metrics)
        best = metrics["best_model"]
        cm = np.array(metrics["results"][best]["test"]["confusion_matrix"])
        plot_confusion(cm, f"Matrice de confusion — {best} (test)", "matrice_confusion_best.png")
        for name, block in metrics["results"].items():
            slug = name.replace(" ", "_").lower()
            plot_confusion(
                np.array(block["test"]["confusion_matrix"]),
                f"Matrice de confusion — {name} (test)",
                f"matrice_confusion_{slug}.png",
            )
    if (MODELS_DIR / "split_paths.joblib").exists():
        export_demo_images()
    print(f"Figures enregistrées dans {FIGURES_DIR}")


if __name__ == "__main__":
    generate_all_figures()
