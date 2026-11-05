"""Génère le notebook livrable du projet."""

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).with_name("classification_objets_routiers.ipynb")


def md(text: str):
    return nbf.v4.new_markdown_cell(text.strip())


def code(text: str):
    return nbf.v4.new_code_cell(text.strip())


def build():
    nb = nbf.v4.new_notebook()
    nb["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "pygments_lexer": "ipython3"},
    }
    cells = []

    cells.append(md("""
# Classification d’objets routiers par Machine Learning
### Projet PACTE — 4e année — Véhicules autonomes

**Objectif.** À partir d’une image d’objet routier, prédire automatiquement sa classe  
(`voiture`, `pieton`, `cycliste`, `moto`, `bus`) et fournir une confiance exploitable du type :

> Image → Voiture (92 %)

Ce notebook couvre les 6 étapes du cahier des charges : acquisition, prétraitement, exploration, développement des modèles, évaluation, application.
"""))

    cells.append(md("""
## 0. Contexte

Les véhicules autonomes fusionnent caméras, LiDAR et radars. Une brique de perception consiste à **classifier** les objets déjà localisés (boîtes englobantes). On se place volontairement sur ce sous-problème : pas de détection, uniquement la reconnaissance de classe, avec des **classifieurs de Machine Learning classique** après extraction de descripteurs (HOG, couleur, LBP), conformément à l’étape 4 du sujet.

**Jeu retenu : Pascal VOC 2007** (crops des classes `car`, `person`, `bicycle`, `motorbike`, `bus`). Justification détaillée dans `data/DESCRIPTION_DATASET.md`.
"""))

    cells.append(md("## 1. Environnement et imports"))

    cells.append(code("""
from pathlib import Path
import sys
import json
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import joblib

ROOT = Path.cwd()
if not (ROOT / "src").exists():
    ROOT = Path.cwd().parent
sys.path.insert(0, str(ROOT))

from src.config import CLASS_TO_IDX, CLASSES, CROPS_DIR, DEMO_DIR, FIGURES_DIR, MODELS_DIR, RANDOM_STATE
from src.dataset import download_voc, extract_crops, load_image_paths
from src.features import extract_many, load_rgb
from src.evaluate import plot_class_distribution, plot_examples
from src.train import make_split_indices

sns.set_theme(style="whitegrid")
%matplotlib inline
print("Racine du projet :", ROOT)
print("Classes :", CLASSES)
"""))

    cells.append(md("""
## 2. Étape 1 — Acquisition des données

Le sous-ensemble Pascal VOC 2007 trainval est téléchargé, puis chaque objet annoté des 5 classes est recadré. Cela simule la sortie d’un détecteur embarqué : le classifieur ne voit que l’objet, pas la scène entière. Le test utilisé ensuite est un split interne, groupé par image source.
"""))

    cells.append(code("""
voc_root = download_voc()
extract_crops(voc_root)
paths, labels = load_image_paths()
print(f"{len(paths)} images chargées")
print(Counter(labels))
"""))

    cells.append(md("""
## 3. Étape 2 — Prétraitement

Pipeline appliqué à chaque crop :

1. **Nettoyage** : objets `difficult` VOC exclus, boîtes < 24 px exclues (signal trop faible).
2. **Redimensionnement** : 64×64 (compromis vitesse / détail pour HOG).
3. **Normalisation** : intensités RGB dans [0, 1] ; plus tard StandardScaler sur les vecteurs.
4. **Séparation** train / validation / test environ 70 / 15 / 15, stratifiée au mieux par classe et groupée par image source pour éviter les fuites.
"""))

    cells.append(code("""
IMAGE_SIZE = (64, 64)
exemple = load_rgb(paths[0], size=IMAGE_SIZE)
print("Tensor image :", exemple.shape, "min/max =", float(exemple.min()), float(exemple.max()))

idx = np.arange(len(labels))
y_all = np.asarray([CLASS_TO_IDX[label] for label in labels], dtype=np.int64)
idx_train, idx_val, idx_test = make_split_indices(paths, y_all)

print(f"Train {len(idx_train)} | Val {len(idx_val)} | Test {len(idx_test)}")
for nom, subset in [("train", idx_train), ("val", idx_val), ("test", idx_test)]:
    print(nom, Counter(labels[subset]))
"""))

    cells.append(md("""
## 4. Étape 3 — Exploration des données
"""))

    cells.append(code("""
df_desc = pd.DataFrame({
    "classe": CLASSES,
    "nombre": [int((labels == c).sum()) for c in CLASSES],
})
df_desc["proportion_%"] = (100 * df_desc["nombre"] / df_desc["nombre"].sum()).round(2)
display(df_desc)

print("Résolution après prétraitement : 64 × 64 × 3")
print("Format : JPEG RGB")
print("Déséquilibre : rapport max/min =", round(df_desc["nombre"].max() / max(df_desc["nombre"].min(), 1), 2))
"""))

    cells.append(code("""
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
plot_class_distribution(df_desc.rename(columns={"nombre": "nombre"}).assign(proportion=df_desc["nombre"] / df_desc["nombre"].sum()))
plot_examples()

fig, axes = plt.subplots(1, 5, figsize=(12, 2.8))
for ax, cls in zip(axes, CLASSES):
    f = next((CROPS_DIR / cls).glob("*.jpg"))
    ax.imshow(Image.open(f))
    ax.set_title(cls)
    ax.axis("off")
plt.suptitle("Un exemple par classe")
plt.show()
"""))

    cells.append(md("""
### Lecture du déséquilibre

Les classes `bus` et `cycliste` sont en général moins peuplées que `pieton` et `voiture` dans VOC. Conséquences :

- un classifieur naïf peut négliger les classes rares ;
- on utilise `class_weight="balanced"` (SVM, Random Forest) et des métriques **macro** (chaque classe compte autant) ;
- le plafond de 700 images par classe évite qu’une classe majoritaire écrase le jeu.
"""))

    cells.append(md("""
## 5. Étape 4 — Extraction de caractéristiques et modèles

Les pixels bruts (64×64×3 = 12 288 dimensions) sont bruités pour un SVM / k-NN. On extrait un vecteur compact :

| Descripteur | Information capturée |
| --- | --- |
| **HOG** | contours / silhouette (piéton vs véhicule) |
| **Histogramme couleur** | teintes (bus souvent saturés, voitures variées) |
| **LBP** | texture locale (carrosserie vs vêtements) |

Les trois classifieurs comparés : **k-NN**, **SVM RBF**, **Random Forest**.  
Chaque modèle est encapsulé dans `StandardScaler → PCA (95 % de variance) → classifieur`.
"""))

    cells.append(code("""
print("Extraction des descripteurs (peut prendre quelques minutes)...")
X = extract_many(paths)
print("Matrice X :", X.shape)

X_train, y_train = X[idx_train], y_all[idx_train]
X_val, y_val = X[idx_val], y_all[idx_val]
X_test, y_test = X[idx_test], y_all[idx_test]
"""))

    cells.append(code("""
def make_pipeline(clf):
    return Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=0.95, random_state=RANDOM_STATE)),
        ("clf", clf),
    ])

modeles = {
    "k-NN": KNeighborsClassifier(n_neighbors=5, weights="distance"),
    "SVM": SVC(kernel="rbf", C=10, gamma="scale", class_weight="balanced", probability=True),
    "Random Forest": RandomForestClassifier(
        n_estimators=300, min_samples_split=4, class_weight="balanced",
        random_state=RANDOM_STATE, n_jobs=-1,
    ),
}

def eval_split(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "cm": confusion_matrix(y_true, y_pred, labels=list(range(len(CLASSES)))),
        "report": classification_report(y_true, y_pred, labels=list(range(len(CLASSES))), target_names=CLASSES, zero_division=0),
    }

resultats = {}
fitted = {}
for nom, clf in modeles.items():
    print(f"\\n===== {nom} =====")
    pipe = make_pipeline(clf)
    pipe.fit(X_train, y_train)
    pred_val = pipe.predict(X_val)
    pred_test = pipe.predict(X_test)
    resultats[nom] = {"val": eval_split(y_val, pred_val), "test": eval_split(y_test, pred_test)}
    fitted[nom] = pipe
    print("VAL ", {k: round(resultats[nom]["val"][k], 3) for k in ["accuracy", "precision", "recall", "f1"]})
    print("TEST", {k: round(resultats[nom]["test"][k], 3) for k in ["accuracy", "precision", "recall", "f1"]})
    print(resultats[nom]["test"]["report"])

best_name = max(resultats, key=lambda n: resultats[n]["val"]["f1"])
print("Meilleur modèle (F1 validation) :", best_name)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
modele_final = make_pipeline(modeles[best_name])
modele_final.fit(np.concatenate([X_train, X_val]), np.concatenate([y_train, y_val]))
joblib.dump(modele_final, MODELS_DIR / "best_model.joblib")
"""))

    cells.append(md("""
## 6. Étape 5 — Évaluation et classes difficiles
"""))

    cells.append(code("""
comp = pd.DataFrame([
    {
        "modèle": nom,
        "Accuracy": r["test"]["accuracy"],
        "Precision": r["test"]["precision"],
        "Recall": r["test"]["recall"],
        "F1-score": r["test"]["f1"],
    }
    for nom, r in resultats.items()
])
display(comp.round(3))

melted = comp.melt(id_vars="modèle", var_name="métrique", value_name="score")
plt.figure(figsize=(8.5, 4.5))
sns.barplot(data=melted, x="métrique", y="score", hue="modèle")
plt.ylim(0, 1)
plt.title("Comparaison des modèles — jeu de test")
plt.show()
"""))

    cells.append(code("""
fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
for ax, (nom, r) in zip(axes, resultats.items()):
    sns.heatmap(r["test"]["cm"], annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASSES, yticklabels=CLASSES, ax=ax)
    ax.set_title(nom)
    ax.set_xlabel("Prédit")
    ax.set_ylabel("Réel")
plt.suptitle("Matrices de confusion (test)")
plt.tight_layout()
plt.show()
"""))

    cells.append(code("""
cm = resultats[best_name]["test"]["cm"]
print(f"Analyse des confusions — modèle {best_name}")
for i, cls in enumerate(CLASSES):
    total = cm[i].sum()
    ok = cm[i, i]
    errs = [(CLASSES[j], int(cm[i, j])) for j in range(len(CLASSES)) if j != i and cm[i, j] > 0]
    errs = sorted(errs, key=lambda t: -t[1])
    print(f"- {cls:10s}  rappel={ok / max(total, 1):.2f}  confusions principales={errs[:3]}")
"""))

    cells.append(md("""
### Interprétation typique (à relier à *votre* matrice)

- **Moto vs cycliste** : silhouettes deux-roues proches (roues, profil étroit). Le HOG les confond si la résolution 64×64 lisse le moteur / le guidon.
- **Bus vs voiture** : même famille « véhicule 4 roues » ; un bus lointain ou partiellement recadré ressemble à une berline.
- **Piéton** : classe souvent la plus peuplée, poses et vêtements très variables ; occlusions fréquentes dans VOC.
- **Voiture** : grande variabilité d’angle (face, profil, arrière) et de couleur.

Ces erreurs sont attendues d’un système de perception : les classes visuellement proches restent les plus difficiles, même avec un détecteur parfait en amont.
"""))

    cells.append(md("""
## 7. Étape 6 — Application : prédiction exploitable

Format imposé : **Image → Classe (xx %)**. Le modèle final est celui de meilleur F1 sur la validation, puis réentraîné sur train+validation. Les images de `data/demo/` proviennent du split test interne (images sources jamais vues à l’entraînement ni à la validation).
"""))

    cells.append(code("""
from src.evaluate import export_demo_images
from src.predict import format_prediction, predict_image

joblib.dump({"train": paths[idx_train], "val": paths[idx_val], "test": paths[idx_test]}, MODELS_DIR / "split_paths.joblib")
demo_files = export_demo_images(n_per_class=2)

print("Modèle final :", best_name, "\\n")
for p in demo_files:
    label, conf, scores = predict_image(p, model=modele_final)
    print(format_prediction(p, label, conf))
"""))

    cells.append(code("""
# Visualisation : image + distribution de probabilités
n = min(6, len(demo_files))
fig, axes = plt.subplots(n, 2, figsize=(8, 2.4 * n))
if n == 1:
    axes = np.array([axes])
for i, p in enumerate(demo_files[:n]):
    label, conf, scores = predict_image(p, model=modele_final)
    axes[i, 0].imshow(Image.open(p))
    axes[i, 0].axis("off")
    axes[i, 0].set_title(format_prediction(p, label, conf))
    axes[i, 1].barh(CLASSES, [scores[c] for c in CLASSES], color="#2a9d8f")
    axes[i, 1].set_xlim(0, 1)
    axes[i, 1].set_xlabel("Probabilité")
plt.tight_layout()
plt.show()
"""))

    cells.append(md("""
## 8. Conclusion

| Étape | Résultat |
| --- | --- |
| Données | Crops Pascal VOC 2007, 5 classes routières, 64×64 |
| Descripteurs | HOG + couleur + LBP, puis PCA |
| Modèles | k-NN, SVM, Random Forest comparés sur métriques macro |
| Décision | meilleur F1 validation, rapporté ensuite sur le test |
| Démo | `Image → Classe (confiance %)` |

**Limites.** Classification hors contexte (pas de tracking, pas de fusion LiDAR). La classe `cycliste` est un proxy (bicyclette VOC). Un CNN (ResNet, MobileNet) améliorerait probablement le rappel des deux-roues, hors périmètre « ML classique » du sujet.

**Pour classifier une image hors notebook :**

```powershell
python -m src.predict chemin\\image.jpg
```
"""))

    nb["cells"] = cells
    OUT.write_text(nbf.writes(nb), encoding="utf-8")
    print("Notebook écrit :", OUT)


if __name__ == "__main__":
    build()
