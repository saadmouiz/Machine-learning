# Classification d’objets routiers par Machine Learning

Projet **PACTE Machine Learning – 4e année**  
Intitulé : *Classification d’objets routiers par Machine Learning pour les véhicules autonomes*

Ce dépôt contient l’ensemble des livrables : notebook, jeu de données (extrait de Pascal VOC 2007), comparaison de classifieurs, rapport, support de présentation et démonstration.

## Classes visées

| Classe projet | Source Pascal VOC | Rôle dans la scène routière |
| --- | --- | --- |
| `voiture` | `car` | véhicules légers |
| `pieton` | `person` | usagers vulnérables |
| `cycliste` | `bicycle` | deux-roues non motorisés |
| `moto` | `motorbike` | deux-roues motorisés |
| `bus` | `bus` | véhicules lourds de transport |

## Structure

```
projet 3ph/
├── notebooks/classification_objets_routiers.ipynb   # traitement complet
├── src/                                             # code réutilisable (démo / pipeline)
├── data/processed/crops/                            # images 64×64 par classe
├── data/demo/                                       # images « nouvelles » pour la démo
├── models/                                          # modèles entraînés + métriques
├── reports/                                         # rapport, figures, présentation
├── scripts/run_all.py                               # pipeline de bout en bout
└── requirements.txt
```

## Installation

```powershell
cd "C:\Users\hp\Desktop\projet 3ph"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Exécution

**1. Pipeline complet (téléchargement VOC, extraction, entraînement, figures, démo)**

```powershell
python -m scripts.run_all
```

Si le module n’est pas trouvé :

```powershell
python scripts/run_all.py
```

**2. Étape par étape**

```powershell
python -m src.dataset      # acquisition + crops
python -m src.train        # k-NN, SVM, Random Forest
python -m src.evaluate     # graphiques et images de démo
python -m src.predict --demo
```

**3. Prédiction sur une image personnelle**

```powershell
python -m src.predict chemin\vers\image.jpg
```

Sortie attendue :

```
image.jpg → Voiture (92 %)
```

**4. Interface web pour importer une image**

Après l’installation des dépendances et l’entraînement du modèle, lancez :

```powershell
python -m streamlit run app.py
```

Le navigateur s’ouvrira sur une interface où vous pourrez déposer une image et consulter la classe prédite ainsi que les scores des cinq classes. Le modèle classe l’image entière fournie ; il ne détecte pas automatiquement plusieurs objets dans une scène.

**5. Notebook**

Ouvrir `notebooks/classification_objets_routiers.ipynb` dans Jupyter ou Cursor, puis exécuter toutes les cellules.

## Choix du jeu de données (à valider avec l’encadrant)

**Pascal VOC 2007** est retenu pour les raisons suivantes :

- jeu académique de référence en perception visuelle ;
- annotations par boîtes englobantes, ce qui simule la sortie d’un détecteur embarqué ;
- les cinq classes du cahier des charges sont présentes ;
- volume exploitable sur un PC étudiant (quelques milliers de crops 64×64).

La description détaillée (provenance, effectifs, résolution, qualité) se trouve dans `data/DESCRIPTION_DATASET.md` et dans le rapport.

## Livrables

| Livrable | Fichier |
| --- | --- |
| Notebook | `notebooks/classification_objets_routiers.ipynb` |
| Description des données | `data/DESCRIPTION_DATASET.md` |
| Comparaison des modèles | `reports/figures/comparaison_modeles.csv` et le notebook |
| Rapport | `reports/Rapport_Projet_PACTE.md` |
| Présentation orale | `reports/Presentation_orale.html` |
| Démonstration | `python -m src.predict --demo` |

## Méthode en bref

1. Découpage des objets VOC → images 64×64  
2. Normalisation, split train / validation / test stratifié  
3. Descripteurs : HOG + histogramme couleur + LBP  
4. Classifieurs : k-NN, SVM (RBF), Random Forest (avec StandardScaler + PCA)  
5. Métriques : accuracy, precision, recall, F1-score, matrices de confusion  
6. Démo : `Image → Classe (confiance %)`
