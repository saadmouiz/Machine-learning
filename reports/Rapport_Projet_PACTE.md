# Classification d’objets routiers par Machine Learning pour les véhicules autonomes

**Projet PACTE — Machine Learning — 4e année**  
Équipe projet — Cycle ingénieur  

---

## 1. Introduction et contexte

Un véhicule autonome doit percevoir son environnement pour planifier une trajectoire sûre. Les capteurs (caméras, LiDAR, radars) alimentent une chaîne dont une étape clé est l’**identification des objets** : voitures, piétons, cyclistes, motos, bus, etc.

Ce projet ne traite pas la détection (localisation dans l’image) mais la **classification** : étant donné une image déjà recadrée autour d’un objet, attribuer une classe et une confiance. C’est exactement le rôle du classifieur qui, dans une architecture réelle, reçoit les boîtes d’un détecteur (YOLO, Faster R-CNN, etc.).

**Objectif.** Concevoir, comparer et évaluer plusieurs classifieurs de Machine Learning classique, puis produire une prédiction exploitable du type `Image → Piéton (87 %)`.

---

## 2. Choix et description des données

### 2.1 Jeu retenu : Pascal VOC 2007

| Critère | Choix |
| --- | --- |
| Corpus source | *PASCAL Visual Object Classes Challenge 2007* |
| Provenance | University of Oxford, corpus public de référence (Everingham et al., IJCV 2010) |
| Usage | extraction des boîtes englobantes → dataset de classification |

**Justification.** Les cinq classes du cahier des charges existent nativement (`car`, `person`, `bicycle`, `motorbike`, `bus`). Les images sont des photographies réelles, avec occlusions, changements d’échelle et d’illumination, ce qui est plus représentatif d’une caméra embarquée qu’un jeu synthétique. Le volume reste compatible avec un PC étudiant, contrairement à KITTI, BDD100K ou nuScenes.

**Alternatives écartées :** GTSRB (panneaux seulement), CIFAR-10/100 (mapping de classes trop approximatif), grands corpus de conduite autonome (coût de stockage et de calcul).

### 2.2 Construction du corpus

1. Téléchargement de l’archive trainval VOC 2007.  
2. Parsing des fichiers XML d’annotation.  
3. Filtrage : classes cibles uniquement, exclusion des objets `difficult`, exclusion des boîtes inférieures à 24×24 pixels.  
4. Recadrage RGB et redimensionnement à **64×64**.  
5. Plafond de **700 images par classe** pour limiter le déséquilibre.

| Classe projet | Classe VOC | Signification |
| --- | --- | --- |
| voiture | car | véhicule léger |
| pieton | person | personne |
| cycliste | bicycle | bicyclette (proxy du cycliste) |
| moto | motorbike | deux-roues motorisé |
| bus | bus | véhicule lourd de transport |

**Limite méthodologique.** VOC annote le vélo et la personne séparément. La classe `cycliste` du projet est donc portée par l’objet *bicycle*. Ce point doit être assumé à l’oral.

Les effectifs exacts après extraction sont consignés dans `reports/figures/description_dataset.csv`.

### 2.3 Qualité et résolution

Les photos VOC sont de qualité amateur / semi-pro, résolution d’origine variable. Après prétraitement, toutes les images sont en **64×64×3**, JPEG. Cette taille conserve une silhouette exploitable par HOG tout en gardant un temps d’apprentissage raisonnable.

---

## 3. Prétraitement et protocole expérimental

### 3.1 Nettoyage, redimensionnement, normalisation

- Nettoyage sémantique (objets difficiles et trop petits).  
- Interpolation bilinéaire vers 64×64.  
- Intensités ramenées à [0, 1] puis **StandardScaler** sur les descripteurs.  
- Réduction de dimension par **PCA** (95 % de variance expliquée) pour stabiliser SVM et k-NN.

### 3.2 Séparation des données

Découpage **groupé par image source et équilibré au mieux par classe** (graine 42) :

- environ 70 % entraînement  
- environ 15 % validation (sélection du modèle)  
- environ 15 % test (évaluation unique, jamais utilisée pour le choix d’hyperparamètres)

Les proportions sont approximatives afin qu’aucune image source ne soit partagée entre les ensembles. Les images de démonstration sont copiées depuis le split test interne : ce sont de **nouvelles** images au sens du modèle.

---

## 4. Exploration

L’exploration (histogrammes, camembert, mosaïque d’exemples) est dans le notebook et dans `reports/figures/`.

On s’attend à un **déséquilibre** : `person` et `car` sont sur-représentés dans VOC par rapport à `bus` et `bicycle`. Mesures prises :

- plafond par classe ;  
- `class_weight="balanced"` ;  
- métriques **macro** (moyenne non pondérée par classe), plus honnêtes que l’accuracy brute.

---

## 5. Descripteurs et modèles

### 5.1 Caractéristiques

| Bloc | Rôle |
| --- | --- |
| HOG (9 orientations, cellules 8×8, blocs 2×2) | géométrie / contours |
| Histogramme couleur (16 bins × 3 canaux) | apparence globale |
| LBP uniforme | texture |

Concaténation puis PCA. Ce choix reste dans le périmètre « ML classique + extraction de caractéristiques » du sujet.

### 5.2 Classifieurs comparés

1. **k-NN** (k = 5, vote pondéré par la distance) — baseline géométrique.  
2. **SVM** à noyau RBF (C = 10, `class_weight=balanced`, probabilités de Platt) — référence historique en vision.  
3. **Random Forest** (300 arbres, `class_weight=balanced`) — modèle ensembliste robuste au bruit.

Tous passent par le même pipeline `Scaler → PCA → classifieur` pour une comparaison équitable.

---

## 6. Évaluation

Les métriques demandées sont calculées sur le **test** :

- Accuracy  
- Precision macro  
- Recall macro  
- F1-score macro  
- Matrice de confusion  

Le **meilleur modèle** est choisi par F1 macro de **validation**, puis on lit ses performances de test (évite de « choisir sur le test »).

Les tableaux et graphiques générés à l’exécution se trouvent dans :

- `models/metrics.json`  
- `reports/figures/comparaison_modeles.csv`  
- `reports/figures/comparaison_modeles.png`  
- `reports/figures/matrice_confusion_*.png`

### 6.1 Classes difficiles (analyse a priori, à confirmer sur la matrice)

| Confusion probable | Explication visuelle |
| --- | --- |
| moto ↔ cycliste | silhouette deux-roues, peu de pixels à 64×64 |
| bus ↔ voiture | même « boîte » véhicule, échelle trompeuse |
| pieton | forte variabilité de pose, vêtements, occlusion |

Ces erreurs illustrent une limite de la perception caméra 2D sans contexte temporel ni LiDAR.

---

## 7. Application

Le classifieur final expose une API simple :

```text
image.jpg → Voiture (92 %)
```

Commande :

```powershell
python -m src.predict --demo
python -m src.predict une_image.jpg
```

La confiance est la probabilité a posteriori du modèle (`predict_proba`). Pour le SVM, il s’agit d’une estimation (Platt), à interpréter comme un score calibré approximatif, pas comme une vraie probabilité bayésienne.

---

## 8. Résultats

Résultats obtenus avec le split groupé par image source : 1 626 images d’entraînement, 345 de validation et 344 de test. La sélection du modèle est faite avec le F1 macro de validation ; le SVM est ensuite retenu. Les métriques finales ci-dessous sont celles du jeu de test.

| Modèle | Accuracy | Precision macro | Recall macro | F1-score macro |
| --- | ---: | ---: | ---: | ---: |
| k-NN | 0,640 | 0,630 | 0,680 | 0,625 |
| SVM RBF | **0,782** | **0,769** | **0,758** | **0,762** |
| Random Forest | 0,645 | 0,666 | 0,594 | 0,613 |

Le SVM obtient le meilleur F1 macro sur validation (0,753), puis 0,762 sur le test. Sur le test, les rappels par classe du SVM sont : voiture 0,84 ; piéton 0,81 ; cycliste 0,71 ; moto 0,74 ; bus 0,70. Le bus est donc la classe au rappel le plus faible pour ce modèle. La matrice montre notamment des confusions entre vélo et moto (10 vélos prédits moto), ainsi qu’entre bus et voiture (6 bus prédits voiture). Ces erreurs sont cohérentes avec les silhouettes proches et le faible nombre d’exemples bus.

Une image externe `voiture.jpg` a aussi été testée dans l’application en ligne de commande : prédiction `voiture`, score estimé 92,5 %. Ce test illustre le fonctionnement de la démo ; il ne remplace pas l’évaluation statistique sur le jeu de test.

Les détails complets (matrices et rapports par classe) sont conservés dans `models/metrics.json`. Les graphiques générés sont dans `reports/figures/`.

---

## 9. Limites et perspectives

- Pas de détection : le système suppose un crop déjà centré.  
- Pas de deep learning (hors sujet PACTE « classifieurs classiques »), alors qu’un CNN léger serait l’état de l’art.  
- Résolution 64×64 limite la discrimination des deux-roues.  
- Classe cycliste = vélo VOC.  
- Une seule graine, pas de validation croisée k-fold (temps de calcul HOG).

**Pistes :** augmenter la résolution (128×128), ajouter un descripteur de forme (moments de Hu), fusionner personne+vélo pour un vrai « cycliste », calibrer les probabilités (isotonic), tester un histogramme d’orientations couleur.

---

## 10. Répartition des livrables

| Livrable | Emplacement |
| --- | --- |
| Notebook | `notebooks/classification_objets_routiers.ipynb` |
| Données + description | `data/` et `data/DESCRIPTION_DATASET.md` |
| Comparaison | notebook + `reports/figures/` |
| Rapport | ce fichier |
| Présentation | `reports/Presentation_orale.html` |
| Démo | `python -m src.predict --demo` |

---

## Référence bibliographique

Everingham, M., Van Gool, L., Williams, C. K. I., Winn, J., & Zisserman, A. (2010). The PASCAL Visual Object Classes (VOC) Challenge. *International Journal of Computer Vision*, 88(2), 303–338.
