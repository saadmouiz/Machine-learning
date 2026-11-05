# Comparaison des modèles

Ce fichier est le livrable « comparaison des modèles ». Les valeurs numériques sont produites à l’exécution (`python run_all.py` ou le notebook) et recopiées dans `reports/figures/comparaison_modeles.csv` ainsi que `models/metrics.json`.

## Protocole commun

Tous les classifieurs reçoivent **le même** vecteur de caractéristiques (HOG + histogramme couleur + LBP), **le même** prétraitement (StandardScaler + PCA 95 %) et **les mêmes** splits stratifiés.

| Modèle | Idée | Hyperparamètres retenus |
| --- | --- | --- |
| k-NN | plus proches voisins | k=5, poids = 1/distance |
| SVM | marge maximale, noyau RBF | C=10, gamma=scale, classes pondérées, probabilités |
| Random Forest | bagging d’arbres | 300 arbres, min_samples_split=4, classes pondérées |

## Critère de sélection

Le modèle **final** est celui du **meilleur F1-score macro sur la validation**.  
Le jeu de test n’est lu qu’ensuite, pour le rapport d’évaluation.

## Tableau à compléter / lu automatiquement

Après entraînement, ouvrir :

- `reports/figures/comparaison_modeles.csv`
- `reports/figures/comparaison_modeles.png`
- `reports/figures/matrice_confusion_best.png`

Colonnes attendues : Accuracy, Precision macro, Recall macro, F1-score macro.

## Lecture recommandée pour le rapport et l’oral

1. Le modèle le plus précis n’est pas forcément le plus utile si une classe rare (bus, cycliste) a un rappel faible.  
2. Commenter la **diagonale** de la matrice et les deux plus grosses cases hors diagonale.  
3. Relier ces cases à la physique de l’image (silhouette, échelle, occlusion), pas seulement au chiffre.
