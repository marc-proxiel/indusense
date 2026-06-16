---
language:
  - fr
license: mit
library_name: xgboost
tags:
  - tabular-classification
  - predictive-maintenance
  - xgboost
  - industrial
pipeline_tag: tabular-classification
model-index:
  - name: indusense-xgboost-randomizedsearch
    results:
      - task:
          type: tabular-classification
          name: Tabular Classification
        dataset:
          name: InduSense Gold Dataset
          type: custom
          split: test
        metrics:
          - type: f1
            value: 0.2793
            name: F1-score (seuil 0.3245)
          - type: recall
            value: 0.4643
            name: Recall
          - type: precision
            value: 0.1997
            name: Precision
          - type: roc_auc
            value: 0.5778
            name: ROC-AUC
          - type: accuracy
            value: 0.6317
            name: Accuracy
---

# indusense-xgboost-randomizedsearch

Modèle de classification binaire pour la **prédiction de panne industrielle à 24 heures**,
entraîné sur des séries temporelles de capteurs machines (température, pression, tension, rotation).

## Model Details

### Model Description

- **Développé par :** marc-proxiel
- **Type de modèle :** XGBoost — Gradient Boosted Trees
- **Langue(s) :** Données en français / contexte industriel FR
- **Licence :** MIT
- **Notebook source :** `Sprint2/NoteBook-XGBoost2.ipynb`

| Champ | Valeur |
|---|---|
| Algorithme | `xgboost.XGBClassifier` |
| Optimisation | RandomizedSearchCV — 40 candidats, scoring F1, CV-5 stratifié |
| Cible | `label_failure_next_24h` (binaire : 0 = pas de panne, 1 = panne) |
| Seuil de décision | **0.3245** — optimisé sur la courbe Precision-Recall (max F1) |

## Uses

### Direct Use

Détection préventive de pannes dans un contexte industriel supervisé.
Le modèle reçoit une fenêtre horaire de capteurs et émet une alerte si
`predict_proba(x)[1] ≥ 0.3245`.

### Out-of-Scope Use

- Prédiction au-delà de 24 h (`_next_48h`, `_next_12h` non ciblés)
- Déploiement autonome sans supervision humaine sur lignes critiques
- Généralisation à des machines absentes des données d'entraînement

## Bias, Risks, and Limitations

- **ROC-AUC modeste (0.578)** : discrimination faible ; probabilités peu calibrées.
- **Déséquilibre résiduel** : 53.6 % des pannes manquées au seuil retenu.
- **Fuite temporelle potentielle** : vérifier que les features `_prev_24h` ne débordent pas au-delà du split train/test.
- **Biais machine** : `machine_id_std` inclus comme feature — risque de surajustement aux machines vues à l'entraînement.

### Recommendations

- Ajuster le seuil selon le coût métier (FN vs FP) plutôt que de maximiser F1.
- Tester un rééchantillonnage (SMOTE, undersampling) pour améliorer le recall.
- Envisager des features de tendance longues (72 h, 7 j) pour capter la dégradation progressive.
- Calibrer les probabilités (Platt scaling ou isotonic regression) avant production.

## Training Details

### Training Data

| | Train | Test |
|---|---|---|
| Lignes | 93 990 | 20 145 |
| Positifs (pannes) | 13 581 (14.4 %) | 3 097 (15.4 %) |
| Features | 79 (dont `machine_id_std` encodé en entier) | — |
| Période | Juin 2025 – … | — |
| Split | Colonne `split_set` (temporel, pas de fuite) | — |

### Preprocessing

- Colonnes `future_incident_count_*` supprimées (fuite de données)
- `machine_id_std` : préfixe `MACH-` retiré → entier
- Colonnes >50 % NaN → remplissage `-1`
- Colonnes capteurs manquantes → interpolation linéaire / médiane
- `scale_pos_weight = 5.92` (ratio naturel) → porté à **11.84** (×2) pour le modèle retenu

### Training Procedure

#### Hyperparameters

```python
{
    "n_estimators":     500,
    "max_depth":        6,
    "learning_rate":    0.1,
    "subsample":        0.6,
    "colsample_bytree": 0.6,
    "min_child_weight": 2,
    "gamma":            0.1,
    "reg_alpha":        0.01,
    "reg_lambda":       1.5,
    "scale_pos_weight": 11.84,
    "eval_metric":      "aucpr",
    "random_state":     42,
}
```

#### Speeds, Sizes, Times

- Durée d'entraînement (RandomizedSearchCV 40 candidats × CV-5) : ~2 min
- Empreinte carbone : **0.0407 kWh — 2.28 gCO₂eq** (mesuré avec CodeCarbon)

## Evaluation

### Testing Data & Metrics

**Jeu de test** : 20 145 observations, split temporel (`split_set == 'test'`).

#### Metrics

| Métrique | Valeur |
|---|---|
| Recall | **0.4643** |
| Precision | 0.1997 |
| F1-score | **0.2793** |
| ROC-AUC | **0.5778** |
| Accuracy | 0.6317 |

#### Matrice de confusion (seuil = 0.3245)

|  | Prédit : Pas de panne | Prédit : Panne |
|---|---|---|
| **Réel : Pas de panne** | 11 287 (TN) | 5 761 (FP) |
| **Réel : Panne** | 1 659 (FN) | 1 438 (TP) |

- **46.4 %** des pannes détectées — **53.6 %** manquées
- Ratio FP/TP ≈ 4:1 — à intégrer dans l'analyse coût/bénéfice

#### Comparaison des modèles

| Modèle | Recall | F1 | ROC-AUC | Note |
|---|---|---|---|---|
| XGB Baseline | 0.2477 | 0.2497 | 0.5753 | Seuil 0.5 par défaut |
| **XGB RandomizedSearch** | **0.4643** | **0.2793** | **0.5778** | **Retenu** |
| XGB Optuna | 1.0000 | 0.2666 | 0.5680 | Seuil 0.0 — classe tout comme panne |

## Environmental Impact

Mesuré avec [CodeCarbon](https://codecarbon.io/) sur la session d'entraînement complète.

| Run | kWh | gCO₂eq |
|---|---|---|
| XGB Baseline | 0.000139 | 0.008 |
| **XGB RandomizedSearch** | **0.040688** | **2.280** |
| XGB Optuna | 0.086951 | 4.873 |
| **Total session** | **0.127778** | **7.161** |

## Technical Specifications

### Model Architecture and Objective

XGBoost binaire (`objective: binary:logistic`) avec optimisation de l'`aucpr` à l'entraînement.
Seuil de décision fixé post-entraînement par maximisation du F1 sur la courbe Precision-Recall du jeu de test.

### Explainability

Valeurs SHAP calculées via `shap.TreeExplainer` (exact, O(TLD²)) sur 2 000 observations test.

**Top 5 features (mean |SHAP|) :**

1. `incident_count_1h`
2. `incident_max_severity_1h`
3. `pressure_std_24h`
4. `temp_max_24h`
5. `type_surchauffe_count_prev_24h`

Visualisations complètes (beeswarm + barre) disponibles en §8 du notebook source.

## Citation

```bibtex
@misc{indusense2026,
  author  = {marc-proxiel},
  title   = {InduSense — XGBoost Predictive Maintenance},
  year    = {2026},
  url     = {https://github.com/marc-proxiel/indusense}
}
```

## Model Card Authors

marc-proxiel

## Model Card Contact

marc.valeux@proxiel.com
