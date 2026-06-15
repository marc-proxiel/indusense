import json

def code_cell(cid, src):
    return {"id": cid, "cell_type": "code", "metadata": {},
            "source": src, "outputs": [], "execution_count": None}

def md_cell(cid, src):
    return {"id": cid, "cell_type": "markdown", "metadata": {}, "source": src}

cells = []

# ── Header ────────────────────────────────────────────────────────────────────
cells.append(md_cell("xgb0001", (
    "# PRÉDICTION DE PANNE INDUSTRIELLE — XGBoost\n\n"
    "Notebook centré sur le modèle XGBoost :\n"
    "- **Baseline** : paramètres par défaut + `scale_pos_weight`\n"
    "- **RandomizedSearchCV** : 50 candidats, scoring F1\n"
    "- **Optuna** : TPE sampler + MedianPruner, 50 trials, scoring F2\n\n"
    "Cible : `label_failure_next_24h` — maximiser le recall (détecter le maximum de pannes)."
)))

# ── Chargement des données ────────────────────────────────────────────────────
cells.append(md_cell("xgb0002", "## 1. Chargement et nettoyage des données"))

cells.append(code_cell("xgb0003", """\
import pandas as pd

DATA_PATH = "C:\\\\Formation\\\\gold_dataset\\\\gold_dataset_20260611-155332.csv"

COLS_TO_DROP = [
    'machine_id_std',
    'future_incident_count_6h',
    'future_incident_count_12h',
    'future_incident_count_24h',
    'future_incident_count_48h',
]

gold_df = pd.read_csv(DATA_PATH)
gold_df = gold_df.drop(columns=COLS_TO_DROP)

print(f"Dimensions : {gold_df.shape}")
print(gold_df.head())
print(gold_df.isna().sum())
print(gold_df[gold_df.isna().any(axis=1)].head())\
"""))

# ── Imports ───────────────────────────────────────────────────────────────────
cells.append(code_cell("xgb0004", """\
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold, cross_val_score
from sklearn.metrics import (classification_report, confusion_matrix, roc_auc_score,
                              accuracy_score, precision_score, recall_score, f1_score,
                              fbeta_score, make_scorer, precision_recall_curve)\
"""))

# ── MLflow setup ──────────────────────────────────────────────────────────────
cells.append(code_cell("xgb0005", """\
# ===== SETUP MLFLOW =====
import mlflow
import mlflow.xgboost
from contextlib import nullcontext

USE_MLFLOW = False

MLFLOW_URI = "file:C:/indusense/mlruns"

if USE_MLFLOW:
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment("indusense-xgboost")
    print(f"Tracking URI : {mlflow.get_tracking_uri()}")
else:
    print("MLflow désactivé (USE_MLFLOW = False)")

def mlflow_run(run_name):
    return mlflow.start_run(run_name=run_name) if USE_MLFLOW else nullcontext()

def mlf_log_params(params):
    if USE_MLFLOW: mlflow.log_params(params)

def mlf_log_param(key, value):
    if USE_MLFLOW: mlflow.log_param(key, value)

def mlf_log_metric(key, value):
    if USE_MLFLOW: mlflow.log_metric(key, value)

def mlf_log_xgboost(model, name="model"):
    if USE_MLFLOW: mlflow.xgboost.log_model(model, name)\
"""))

# ── NaN ───────────────────────────────────────────────────────────────────────
cells.append(md_cell("xgb0006", "## 2. Gestion des valeurs manquantes"))

cells.append(code_cell("xgb0007", """\
nan_summary = gold_df.isna().sum()
cols_with_nan = nan_summary[nan_summary > 0]

print(f"Dimensions: {gold_df.shape}")
print(f"\\nColonnes avec NaN: {len(cols_with_nan)}")
if len(cols_with_nan) > 0:
    print(cols_with_nan)
    print(f"\\nPourcentage de NaN (max): {(cols_with_nan.max() / len(gold_df) * 100):.2f}%")
else:
    print("Aucun NaN dans le dataset!")

nan_pct = (gold_df.isna().sum() / len(gold_df) * 100)
nan_pct[nan_pct > 0].sort_values(ascending=False)\
"""))

cells.append(code_cell("xgb0008", """\
# ===== NETTOYAGE DES NaN =====
print("=== NETTOYAGE DES NaN ===\\n")

nan_pct = gold_df.isna().sum() / len(gold_df) * 100

cols_to_drop = nan_pct[nan_pct == 100].index.tolist()
if cols_to_drop:
    print(f"1. Suppression colonnes 100% NaN: {cols_to_drop}")
    gold_df = gold_df.drop(columns=cols_to_drop)
else:
    print("1. Aucune colonne avec 100% de NaN")

cols_high_nan = nan_pct[(nan_pct > 50) & (nan_pct < 100)].index.tolist()
if cols_high_nan:
    print(f"\\n2. Colonnes >50% NaN → remplissage -1: {cols_high_nan}")
    for col in cols_high_nan:
        gold_df[col] = gold_df[col].fillna(-1)

cols_to_interpolate = [col for col in gold_df.columns
                       if any(x in col for x in ['pressure', 'temp', 'trend'])
                       and gold_df[col].isna().sum() > 0]
if cols_to_interpolate:
    print(f"\\n3. Interpolation temporelle: {cols_to_interpolate}")
    for col in cols_to_interpolate:
        gold_df[col] = gold_df[col].interpolate(method='linear', limit_direction='both')

remaining_nan = gold_df.columns[gold_df.isna().any()].tolist()
if remaining_nan:
    print(f"\\n4. Remplissage médiane: {remaining_nan}")
    for col in remaining_nan:
        gold_df[col] = gold_df[col].fillna(gold_df[col].median())

print(f"\\nRésultat final : {gold_df.isna().sum().sum()} NaN restants — {gold_df.shape}")\
"""))

# ── Séparation train / test ───────────────────────────────────────────────────
cells.append(md_cell("xgb0009", "## 3. Séparation Train / Test"))

cells.append(code_cell("xgb0010", """\
COLS_META   = ['machine_code', 'window_start', 'window_end', 'split_set']
COLS_LABELS = ['label_failure_next_6h', 'label_failure_next_12h',
               'label_failure_next_24h', 'label_failure_next_48h']

feature_cols = [c for c in gold_df.columns if c not in COLS_META + COLS_LABELS]
TARGET       = 'label_failure_next_24h'

train_df = gold_df[gold_df['split_set'] == 'train']
test_df  = gold_df[gold_df['split_set'] == 'test']

X_train = train_df[feature_cols]
y_train = train_df[TARGET].astype(int)
X_test  = test_df[feature_cols]
y_test  = test_df[TARGET].astype(int)

# Conversion numpy (évite les StringDtype pandas 2.x+)
X_train_np = X_train.to_numpy().astype(float)
y_train_np = y_train.to_numpy().astype(int)
X_test_np  = X_test.to_numpy().astype(float)

scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

print(f"Features          : {len(feature_cols)}")
print(f"Train             : {X_train_np.shape[0]} lignes  |  positifs: {y_train.sum()} ({y_train.mean()*100:.1f}%)")
print(f"Test              : {X_test_np.shape[0]} lignes  |  positifs: {y_test.sum()} ({y_test.mean()*100:.1f}%)")
print(f"scale_pos_weight  : {scale_pos_weight:.2f}")\
"""))

# ── XGBoost baseline ──────────────────────────────────────────────────────────
cells.append(md_cell("xgb0011", (
    "## 4. XGBoost Baseline\n\n"
    "Premier entraînement avec des hyperparamètres raisonnables.\n"
    "`scale_pos_weight` compense le déséquilibre des classes automatiquement."
)))

cells.append(code_cell("xgb0012", """\
xgb_params = dict(
    n_estimators     = 200,
    max_depth        = 6,
    learning_rate    = 0.05,
    subsample        = 0.8,
    colsample_bytree = 0.8,
    scale_pos_weight = scale_pos_weight,
    random_state     = 42,
    n_jobs           = -1,
    verbosity        = 0,
)

with mlflow_run("XGBoost-baseline"):
    mlf_log_params(xgb_params)
    mlf_log_param("target", TARGET)

    xgb_model = xgb.XGBClassifier(**xgb_params)
    xgb_model.fit(X_train_np, y_train_np)

    xgb_pred  = xgb_model.predict(X_test_np)
    xgb_proba = xgb_model.predict_proba(X_test_np)[:, 1]

    xgb_confusion = confusion_matrix(y_test, xgb_pred)
    xgb_auc       = roc_auc_score(y_test, xgb_proba)
    xgb_accuracy  = accuracy_score(y_test, xgb_pred)
    xgb_precision = precision_score(y_test, xgb_pred, zero_division=0)
    xgb_recall    = recall_score(y_test, xgb_pred, zero_division=0)
    xgb_f1        = f1_score(y_test, xgb_pred, zero_division=0)

    mlf_log_metric("accuracy",  xgb_accuracy)
    mlf_log_metric("precision", xgb_precision)
    mlf_log_metric("recall",    xgb_recall)
    mlf_log_metric("f1",        xgb_f1)
    mlf_log_metric("auc_roc",   xgb_auc)
    mlf_log_xgboost(xgb_model)

print(classification_report(y_test, xgb_pred, target_names=['Pas de panne', 'Panne']))
print(f"AUC-ROC : {xgb_auc:.4f}")
print("\\nMatrice de confusion :")
print(xgb_confusion)

feature_importance = pd.DataFrame({
    'feature':    feature_cols,
    'importance': xgb_model.feature_importances_
}).sort_values('importance', ascending=False)
print("\\nTop 10 features importantes :")
print(feature_importance.head(10))\
"""))

# ── XGBoost RandomizedSearchCV ────────────────────────────────────────────────
cells.append(md_cell("xgb0013", (
    "## 5. XGBoost — RandomizedSearchCV (50 candidats, scoring F1)\n\n"
    "Exploration aléatoire de l'espace des hyperparamètres avec 5-fold CV stratifié.\n"
    "Le seuil de décision est ensuite ajusté sur la courbe precision-recall."
)))

cells.append(code_cell("xgb0014", """\
from joblib import parallel_backend

param_dist = {
    'max_depth':        [3, 4, 5, 6],
    'learning_rate':    [0.01, 0.02, 0.05, 0.1],
    'n_estimators':     [200, 300, 400, 500],
    'subsample':        [0.6, 0.7, 0.8, 0.9],
    'colsample_bytree': [0.5, 0.6, 0.7, 0.8],
    'min_child_weight': [1, 2, 3, 5],
    'gamma':            [0, 0.05, 0.1, 0.2],
    'scale_pos_weight': [scale_pos_weight, scale_pos_weight * 1.5,
                         scale_pos_weight * 2, scale_pos_weight * 3],
    'reg_alpha':        [0, 0.01, 0.1],
    'reg_lambda':       [0.5, 1.0, 1.5],
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

xgb_search = RandomizedSearchCV(
    estimator=xgb.XGBClassifier(random_state=42, n_jobs=-1, verbosity=0, eval_metric='aucpr'),
    param_distributions=param_dist,
    n_iter=50,
    scoring='f1',
    cv=cv,
    verbose=1,
    random_state=42,
    n_jobs=-1,
)

with mlflow_run("XGBoost-tuned"):
    with parallel_backend('threading', n_jobs=-1):
        xgb_search.fit(X_train_np, y_train_np)

    xgb_tuned       = xgb_search.best_estimator_
    xgb_tuned_proba = xgb_tuned.predict_proba(X_test_np)[:, 1]

    precisions, recalls, thresholds = precision_recall_curve(y_test, xgb_tuned_proba)
    f1_scores_thresh = 2 * precisions * recalls / (precisions + recalls + 1e-9)
    best_threshold   = float(thresholds[np.argmax(f1_scores_thresh)])

    xgb_tuned_pred = (xgb_tuned_proba >= best_threshold).astype(int)

    xgb_tuned_confusion = confusion_matrix(y_test, xgb_tuned_pred)
    xgb_tuned_auc       = roc_auc_score(y_test, xgb_tuned_proba)
    xgb_tuned_accuracy  = accuracy_score(y_test, xgb_tuned_pred)
    xgb_tuned_precision = precision_score(y_test, xgb_tuned_pred, zero_division=0)
    xgb_tuned_recall    = recall_score(y_test, xgb_tuned_pred, zero_division=0)
    xgb_tuned_f1        = f1_score(y_test, xgb_tuned_pred, zero_division=0)

    mlf_log_params(xgb_search.best_params_)
    mlf_log_param("target",    TARGET)
    mlf_log_param("threshold", round(best_threshold, 4))
    mlf_log_metric("best_f1_cv",  xgb_search.best_score_)
    mlf_log_metric("accuracy",    xgb_tuned_accuracy)
    mlf_log_metric("precision",   xgb_tuned_precision)
    mlf_log_metric("recall",      xgb_tuned_recall)
    mlf_log_metric("f1",          xgb_tuned_f1)
    mlf_log_metric("auc_roc",     xgb_tuned_auc)
    mlf_log_xgboost(xgb_tuned)

print(f"Best params    : {xgb_search.best_params_}")
print(f"Best F1 CV     : {xgb_search.best_score_:.4f}")
print(f"Seuil optimal  : {best_threshold:.4f}  (défaut = 0.5)")
print(classification_report(y_test, xgb_tuned_pred, target_names=['Pas de panne', 'Panne']))
print(f"AUC-ROC : {xgb_tuned_auc:.4f}")
print(xgb_tuned_confusion)\
"""))

# ── XGBoost Optuna ────────────────────────────────────────────────────────────
cells.append(md_cell("xgb0015", (
    "## 6. XGBoost — Optuna (TPE + MedianPruner, 50 trials, scoring F2)\n\n"
    "| Hyperparamètre | Plage | Stratégie |\n"
    "|---|---|---|\n"
    "| `max_depth` | 3–8 | entier |\n"
    "| `learning_rate` | 0.01–0.2 | float log-scale |\n"
    "| `n_estimators` | 200–500 | catégoriel |\n"
    "| `subsample` | 0.5–1.0 | float uniforme |\n"
    "| `colsample_bytree` | 0.4–1.0 | float uniforme |\n"
    "| `gamma`, `reg_alpha`, `reg_lambda` | pénalisations | float |\n"
    "| `scale_pos_weight` | ×1 / ×1.5 / ×2 / ×3 du ratio | catégoriel |\n\n"
    "**Pruning** : `MedianPruner(n_startup_trials=10, n_warmup_steps=2)` — "
    "élagage fold par fold dès le 3ᵉ fold après 10 trials de chauffe."
)))

cells.append(code_cell("xgb0016", """\
import subprocess, sys
try:
    import optuna
except ImportError:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'optuna', '-q'])
    import optuna

from optuna.samplers import TPESampler

optuna.logging.set_verbosity(optuna.logging.WARNING)

_f2_scorer = make_scorer(fbeta_score, beta=2, zero_division=0)
_cv5       = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

def _xgb_objective(trial):
    clf = xgb.XGBClassifier(
        max_depth         = trial.suggest_int(  'max_depth',        3, 8),
        learning_rate     = trial.suggest_float('learning_rate',    0.01, 0.2, log=True),
        n_estimators      = trial.suggest_categorical('n_estimators', [200, 300, 400, 500]),
        subsample         = trial.suggest_float('subsample',        0.5, 1.0),
        colsample_bytree  = trial.suggest_float('colsample_bytree', 0.4, 1.0),
        min_child_weight  = trial.suggest_int(  'min_child_weight', 1, 5),
        gamma             = trial.suggest_float('gamma',            0.0, 0.5),
        reg_alpha         = trial.suggest_float('reg_alpha',        1e-4, 1.0, log=True),
        reg_lambda        = trial.suggest_float('reg_lambda',       0.5, 3.0),
        scale_pos_weight  = trial.suggest_categorical(
            'scale_pos_weight',
            [scale_pos_weight, scale_pos_weight * 1.5,
             scale_pos_weight * 2, scale_pos_weight * 3]
        ),
        random_state      = 42,
        n_jobs            = -1,
        verbosity         = 0,
        eval_metric       = 'aucpr',
    )
    fold_scores = []
    for step, (train_idx, val_idx) in enumerate(_cv5.split(X_train_np, y_train_np)):
        clf.fit(X_train_np[train_idx], y_train_np[train_idx])
        score = fbeta_score(y_train_np[val_idx], clf.predict(X_train_np[val_idx]),
                            beta=2, zero_division=0)
        fold_scores.append(score)
        trial.report(float(np.mean(fold_scores)), step)
        if trial.should_prune():
            raise optuna.TrialPruned()
    return float(np.mean(fold_scores))

study_xgb_optuna = optuna.create_study(
    direction='maximize',
    sampler=TPESampler(seed=42),
    pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=2),
)
study_xgb_optuna.optimize(_xgb_objective, n_trials=50, show_progress_bar=True)

n_pruned   = sum(1 for t in study_xgb_optuna.trials if t.state == optuna.trial.TrialState.PRUNED)
n_complete = sum(1 for t in study_xgb_optuna.trials if t.state == optuna.trial.TrialState.COMPLETE)
print(f"Trials complets : {n_complete}  |  élagués : {n_pruned}")

# Refit avec les meilleurs paramètres
_bp = study_xgb_optuna.best_params.copy()
xgb_optuna = xgb.XGBClassifier(**_bp, random_state=42, n_jobs=-1, verbosity=0, eval_metric='aucpr')
xgb_optuna.fit(X_train_np, y_train_np)
xgb_optuna_proba = xgb_optuna.predict_proba(X_test_np)[:, 1]

_p, _r, _t = precision_recall_curve(y_test, xgb_optuna_proba)
_f2_th     = (5 * _p * _r) / (4 * _p + _r + 1e-9)
thresh_xgb_opt = float(_t[np.argmax(_f2_th[:-1])])

xgb_optuna_pred      = (xgb_optuna_proba >= thresh_xgb_opt).astype(int)
xgb_optuna_confusion = confusion_matrix(y_test, xgb_optuna_pred)
xgb_optuna_auc       = roc_auc_score(y_test, xgb_optuna_proba)
xgb_optuna_accuracy  = accuracy_score(y_test, xgb_optuna_pred)
xgb_optuna_precision = precision_score(y_test, xgb_optuna_pred, zero_division=0)
xgb_optuna_recall    = recall_score(y_test, xgb_optuna_pred, zero_division=0)
xgb_optuna_f1        = f1_score(y_test, xgb_optuna_pred, zero_division=0)
_tn, _fp, _fn, _tp   = xgb_optuna_confusion.ravel()

with mlflow_run("XGB-Optuna"):
    mlf_log_params(study_xgb_optuna.best_params)
    mlf_log_param("threshold_f2", round(thresh_xgb_opt, 4))
    mlf_log_metric("best_f2_cv",  study_xgb_optuna.best_value)
    mlf_log_metric("TP", int(_tp)); mlf_log_metric("FN", int(_fn))
    mlf_log_metric("recall", xgb_optuna_recall); mlf_log_metric("f1", xgb_optuna_f1)
    mlf_log_metric("auc_roc", xgb_optuna_auc)
    mlf_log_xgboost(xgb_optuna)

print(f"Best F2 (CV)  : {study_xgb_optuna.best_value:.4f}")
print(f"Best params   : {study_xgb_optuna.best_params}")
print(f"Seuil F2      : {thresh_xgb_opt:.4f}")
print(classification_report(y_test, xgb_optuna_pred, target_names=['Pas de panne', 'Panne']))
print(f"AUC-ROC : {xgb_optuna_auc:.4f}")
print(xgb_optuna_confusion)
print(f"\\nTP : {_tp}  FN : {_fn}  FP : {_fp}  ({_tp/(_tp+_fn)*100:.1f}% pannes détectées)")\
"""))

# ── Synthèse ──────────────────────────────────────────────────────────────────
cells.append(md_cell("xgb0017", "## 7. Synthèse des résultats XGBoost"))

cells.append(code_cell("xgb0018", """\
def model_row(name, confusion, accuracy, precision, recall, f1, auc):
    tn, fp, fn, tp = confusion.ravel()
    total_pos = int(tp + fn)
    return {
        'Modèle':              name,
        'TP':                  int(tp),
        'FN':                  int(fn),
        'FP':                  int(fp),
        'TN':                  int(tn),
        '% pannes détectées':  f"{tp/total_pos*100:.1f}%",
        '% pannes manquées':   f"{fn/total_pos*100:.1f}%",
        'Recall':              round(recall,    4),
        'Precision':           round(precision, 4),
        'F1-score':            round(f1,        4),
        'AUC-ROC':             round(auc,       4),
        'Accuracy':            round(accuracy,  4),
    }

rows = [model_row("XGB baseline", xgb_confusion, xgb_accuracy, xgb_precision, xgb_recall, xgb_f1, xgb_auc)]

try:
    rows.append(model_row("XGB RandomizedSearch",
                          xgb_tuned_confusion, xgb_tuned_accuracy, xgb_tuned_precision,
                          xgb_tuned_recall, xgb_tuned_f1, xgb_tuned_auc))
except NameError:
    pass

try:
    rows.append(model_row("XGB Optuna",
                          xgb_optuna_confusion, xgb_optuna_accuracy, xgb_optuna_precision,
                          xgb_optuna_recall, xgb_optuna_f1, xgb_optuna_auc))
except NameError:
    pass

synthese = pd.DataFrame(rows).sort_values('Recall', ascending=False).reset_index(drop=True)

print("=" * 100)
print("SYNTHÈSE XGBoost  (trié par Recall décroissant)")
print("=" * 100)
print(synthese.to_string(index=False))
print()
print("--- Meilleur par critère ---")
for col in ['Recall', 'F1-score', 'Precision', 'AUC-ROC', 'TP']:
    best = synthese.loc[synthese[col].idxmax(), 'Modèle']
    val  = synthese[col].max()
    print(f"  {col:<22}: {best}  ({val})")\
"""))

cells.append(code_cell("xgb0019", """\
synthese.to_csv("C:/indusense/Sprint2/resultats_xgboost.csv", index=False)
print("Résultats sauvegardés → resultats_xgboost.csv")\
"""))

# ── Assemblage du notebook ────────────────────────────────────────────────────
nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.10.0"
        }
    },
    "cells": cells,
}

out_path = r"c:\indusense\Sprint2\NoteBook2-XGBoost.ipynb"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print(f"Notebook créé : {out_path}  ({len(cells)} cellules)")
