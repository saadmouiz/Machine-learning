# Description du jeu de données

## Choix retenu

**Nom :** Pascal VOC 2007 (crops d’objets routiers)  
**Type :** classification d’images à 5 classes, construite à partir d’un jeu de détection.

## Justification (à valider avec l’encadrant)

Le cahier des charges demande des images d’objets routiers (voiture, piéton, cycliste, moto, bus) et non un jeu de détection complet. Pascal VOC 2007 est un corpus public, largement cité, qui contient exactement ces catégories sous forme de boîtes englobantes. En extraire les crops, on obtient un problème de **classification** réaliste : on se place *après* un détecteur (comme dans une pile de perception de véhicule autonome).

Alternatives écartées :

| Jeu | Motif d’écart |
| --- | --- |
| GTSRB | Panneaux uniquement, pas les usagers de la route |
| CIFAR-10 / 100 | Pas de correspondance fidèle aux 5 classes, images 32×32 trop pauvres |
| KITTI / BDD100K / nuScenes | Trop lourds pour un projet PACTE sur PC, licence / volume |
| ImageNet complet | Bruit sémantique, téléchargement disproportionné |

## Provenance

- **Éditeur :** University of Oxford — PASCAL Visual Object Classes Challenge 2007  
- **Miroirs configurés :** `data.brainchip.com`, `host.robots.ox.ac.uk` et `pjreddie.com`
- **Licence d’usage académique :** jeu public de benchmark ; citer Everingham et al., *The PASCAL Visual Object Classes Challenge*, IJCV 2010  
- **Archive téléchargée par le pipeline :** `VOCtrainval_06-Nov-2007.tar`

Le protocole expérimental utilise un découpage interne train/validation/test sur ce sous-ensemble. Les objets recadrés d’une même image source restent dans le même split.

Les images d’origine sont des scènes photographiques naturelles (extérieur, différentes saisons, occlusions, échelles).

## Construction du corpus de classification

Pour chaque annotation XML :

1. conserver uniquement `car`, `person`, `bicycle`, `motorbike`, `bus` ;
2. ignorer les objets marqués `difficult` ;
3. ignorer les boîtes plus petites que 24×24 pixels (trop peu d’information) ;
4. recadrer, convertir en RGB, redimensionner en **64×64** ;
5. limiter à **700 images par classe** afin de limiter le déséquilibre et le temps de calcul.

### Mapping des classes

| Classe livrable | Identifiant VOC | Commentaire |
| --- | --- | --- |
| voiture | car | voitures particulières |
| pieton | person | personnes (piétons, parfois passagers visibles) |
| cycliste | bicycle | bicyclette (proxy du cycliste ; VOC annote le vélo séparément de la personne) |
| moto | motorbike | motocyclettes |
| bus | bus | bus / autocars |

Limite connue, à mentionner à l’oral : la classe `cycliste` est portée par l’objet *bicycle*, pas par le couple personne+vélo. C’est un choix classique et discutable, cohérent avec un classifieur d’objets détectés.

## Caractéristiques des images

| Propriété | Valeur |
| --- | --- |
| Format | JPEG RGB |
| Résolution après prétraitement | 64 × 64 pixels |
| Résolution d’origine | variable (photos VOC, typiquement quelques centaines de pixels de côté) |
| Qualité | photographies réelles, parfois floues, occluses, contre-jour |
| Normalisation | pixels dans [0, 1] puis StandardScaler sur les descripteurs |

Effectifs obtenus après extraction : **2 315 crops** au total, répartis en voiture 700, piéton 700, cycliste 350, moto 338 et bus 227. Les classes voiture et piéton sont plafonnées à 700 ; les autres restent moins représentées, ce qui conserve un déséquilibre à prendre en compte dans l’analyse.

## Organisation sur le disque

```
data/
├── raw/VOCdevkit/VOC2007/     # corpus original (JPEGImages + Annotations)
├── processed/crops/
│   ├── voiture/
│   ├── pieton/
│   ├── cycliste/
│   ├── moto/
│   └── bus/
└── demo/                      # images tenues à l’écart (split test) pour la démo
```

## Séparation

Découpage **groupé par image source et équilibré au mieux par classe** (les proportions peuvent varier légèrement, car plusieurs objets peuvent provenir d’une même image) :

- 70 % entraînement  
- 15 % validation (choix du meilleur modèle)  
- 15 % test (évaluation finale, jamais utilisée pour ajuster les hyperparamètres)

Les chemins du split test interne copiés dans `data/demo/` jouent le rôle des « nouvelles images » demandées à l’étape 6. Aucune image source de démonstration n’est présente dans train ou validation.
