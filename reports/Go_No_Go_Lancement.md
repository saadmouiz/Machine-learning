# Note de décision Go / No-Go — choix du jeu de données

**Projet :** Classification d’objets routiers par Machine Learning pour les véhicules autonomes  
**Cadre :** Projet PACTE Machine Learning — 4e année  
**Date :** 5 novembre 2026  
**Périmètre de cette note :** décision sur le jeu de données et état d’avancement partiel. Ce document n’est pas le rapport final du projet.

## 1. Décision proposée

# GO conditionnel — Pascal VOC 2007 (trainval)

Le jeu de données est adapté pour lancer un prototype de **classification d’objets déjà recadrés** en cinq classes : voiture, piéton, vélo (proxy de cycliste), moto et bus. Il permet de mettre en œuvre les étapes de prétraitement, d’extraction de caractéristiques et de comparaison des classifieurs classiques prévues par le sujet.

La décision reste conditionnée à la validation du choix par l’encadrant, notamment pour l’interprétation de la classe « cycliste » et l’usage d’un split interne plutôt que du test officiel VOC.

## 2. Justification du choix

- **Correspondance avec le sujet :** Pascal VOC 2007 contient les catégories `car`, `person`, `bicycle`, `motorbike` et `bus`.
- **Annotations exploitables :** les boîtes englobantes permettent d’extraire des images d’objets, sans devoir annoter le corpus à la main.
- **Taille raisonnable :** le sous-ensemble trainval est téléchargeable et exploitable sur un ordinateur de projet.
- **Pertinence pédagogique :** le corpus permet une comparaison de modèles de Machine Learning classique après extraction de caractéristiques.

### Limites à accepter explicitement

1. VOC est un jeu généraliste de photos, et non un corpus conçu spécifiquement pour la conduite autonome.
2. La classe `bicycle` désigne le vélo, pas nécessairement un cycliste complet (personne et vélo ensemble). « Cycliste » est donc un proxy.
3. Le pipeline travaille sur des crops d’objets : il ne détecte pas les objets dans une image de scène.
4. Le protocole actuel crée un jeu de test interne depuis trainval ; il ne s’agit pas du test officiel VOC.
5. Les effectifs restent déséquilibrés ; une seule graine de séparation ne suffit pas à démontrer une performance généralisable.
6. Les conditions exactes de licence et de redistribution des données doivent être vérifiées selon les règles de l’établissement avant diffusion du corpus ou du projet.

## 3. État d’avancement partiel

| Étape | État observé |
|---|---|
| Acquisition | Archive VOC 2007 trainval téléchargée et extraite. |
| Prétraitement | Filtrage des objets difficiles et petits, recadrage des boîtes, conversion RGB et redimensionnement à 64 × 64. |
| Corpus obtenu | 2 315 crops : 700 voitures, 700 piétons, 350 vélos, 338 motos et 227 bus. |
| Caractéristiques | Descripteurs HOG, histogrammes couleur et LBP implémentés. |
| Première comparaison | k-NN, SVM RBF et Random Forest entraînés sur le split groupé interne. |
| Résultat indicatif | Le SVM a été retenu par F1 macro validation (0,753) ; sur le test interne : accuracy 0,782 et F1 macro 0,762. |
| Démonstration | Une interface permet de fournir une image et d’afficher une classe prédite ainsi que des scores estimés. Une photo externe de voiture a été prédite « voiture » avec un score estimé de 92,5 %. |

Ces chiffres sont des résultats préliminaires du pipeline en l’état. Le test d’une seule image externe est une démonstration fonctionnelle, pas une validation statistique sur des données nouvelles.

## 4. Conditions avant validation définitive du choix

- Faire confirmer par l’encadrant que VOC 2007 et le cadrage « classification après détection » conviennent au sujet.
- Faire confirmer que `bicycle` peut représenter la classe projet « cycliste » ; sinon, redéfinir cette classe ou choisir un corpus mieux adapté.
- Confirmer l’acceptation d’un split interne groupé par image source, distinct du test officiel VOC.
- Vérifier les conditions de citation, d’utilisation et de redistribution du jeu de données.
- Dans la suite du projet, analyser les performances par classe et tester plusieurs images externes, avec prudence sur le sens des scores de confiance.

## 5. Conclusion du jalon

**Décision : GO conditionnel.** Le corpus retenu permet de démarrer et un premier pipeline fonctionnel existe déjà. Le principal point à faire arbitrer est la représentativité de la classe « cycliste » ; les autres limites doivent être présentées comme celles d’un prototype académique de classification d’objets recadrés, et non d’un système complet de perception pour véhicule autonome.

**Références de suivi :** [description du jeu de données](../data/DESCRIPTION_DATASET.md), [rapport détaillé du projet](Rapport_Projet_PACTE.md), [métriques produites](../models/metrics.json).
