"""
estructura planteada:

detect_drift: Detecta el drift en los datos de entrenamiento (opcional si alcanzamos)
optimize_hyperparameters: Optimiza los hiperparámetros del modelo
train_model: Entrena el modelo con los datos procesados y las características generadas
evaluate_model: Trackear los resultados con MLflow
export_model: Exporta el modelo entrenado para su uso posterior

"""

import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
from optuna.integration import XGBoostPruningCallback
import xgboost as xgb
from sklearn.metrics import f1_score
import numpy as np
from sklearn.metrics import classification_report
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import pandas as pd
import mlflow
import joblib
import json
import os

def detect_drift(train_data):
    pass

# TO-DO: agregar mlflow para trackear los resultados
def optimize_hyperparameters(**kwargs):

    execution_date = kwargs['ds']

    # Load the data
    X_train_tr = pd.read_parquet(os.path.join(execution_date, "data_transformed/X_train.parquet"))
    y_train = pd.read_parquet(os.path.join(execution_date, "data_transformed/y_train.parquet"))["label"]

    X_val_tr = pd.read_parquet((os.path.join(execution_date, "data_transformed/X_val.parquet")))
    y_val = pd.read_parquet((os.path.join(execution_date, "data_transformed/y_val.parquet"))["label"])

    dtrain = xgb.DMatrix(X_train_tr, label=y_train)
    dval   = xgb.DMatrix(X_val_tr,   label=y_val)

    # Definir función objetivo para Optuna
    def objective(trial):
        param = {
            "learning_rate":    trial.suggest_loguniform("learning_rate",    0.10, 0.25),
            "max_depth":        trial.suggest_int("max_depth",               8,   16),
            "subsample":        trial.suggest_float("subsample",             0.65, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree",      0.6,  1.0),
            "gamma":            trial.suggest_float("gamma",                 0.0,  2.0),
            "min_child_weight": trial.suggest_int("min_child_weight",        1,   10),
            "reg_alpha":        trial.suggest_loguniform("reg_alpha",       0.1,   5.0),
            "reg_lambda":       trial.suggest_loguniform("reg_lambda",      0.1,   5.0),
            "tree_method":      "hist",
            "verbosity":        0,
            "eval_metric":      "logloss",
            "objective":        "binary:logistic"
        }
        n_rounds = trial.suggest_int("n_estimators", 50, 300)

        pruning_callback = XGBoostPruningCallback(trial, "validation-logloss")

        booster = xgb.train(
            params=param,
            dtrain=dval,
            num_boost_round=n_rounds,
            evals=[(dval, "validation")],
            callbacks=[
                pruning_callback,
                xgb.callback.EarlyStopping(rounds=30)
            ],
            verbose_eval=False,
        )

        preds = (booster.predict(dval) > 0.5).astype(int)
        return f1_score(y_val, preds, pos_label=1)

    # Ejecutar estudio Optuna
    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(seed=42),
        pruner=MedianPruner(n_startup_trials=5)
    )
    study.optimize(objective, timeout=300)

    # Resultados
    best = study.best_trial
    params_final = best.params.copy()
    params_final.update({
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "verbosity": 0,
        "tree_method": "hist"
    })
    n_rounds_final = best.params["n_estimators"]

    # Búsqueda de umbral óptimo en validación
    booster = xgb.train(
        params=params_final,
        dtrain=dtrain,
        num_boost_round=n_rounds_final,
        evals=[(dval, "validation")],
        early_stopping_rounds=30,
        verbose_eval=False
    )

    probs_val = booster.predict(dval)
    best_thr, best_f1 = 0.0, 0.0
    for thr in np.linspace(0.1, 0.9, 81):
        preds_thr = (probs_val > thr).astype(int)
        f1 = f1_score(y_val, preds_thr, pos_label=1)
        if f1 > best_f1:
            best_f1, best_thr = f1, thr

    # guardar los resultados
    os.makedirs(os.path.join(execution_date, "train"), exist_ok=True)
    with open(os.path.join(execution_date, "train/optuna_params.json"), "w") as f:
        json.dump(params_final, f)


def train_model(**kwargs):
    
    execution_date = kwargs['ds']
    # Cargar datos
    X_train_tr = pd.read_parquet(os.path.join(execution_date, "data_transformed/X_train.parquet"))
    y_train = pd.read_parquet((os.path.join(execution_date, "data_transformed/y_train.parquet")))["label"]

    X_test = pd.read_parquet((os.path.join(execution_date, "data_transformed/X_test.parquet")))
    y_test = pd.read_parquet((os.path.join(execution_date, "data_transformed/y_test.parquet")))["label"]

    dtrain = xgb.DMatrix(X_train_tr, label=y_train)
    dtest = xgb.DMatrix(X_test)

    # Cargar parámetros de Optuna
    with open(os.path.join(execution_date, "train/optuna_params.json"), "r") as f:
        parameters = json.load(f)

    # Entrenar booster final
    booster = xgb.train(
        params=parameters,
        dtrain=dtrain,
        num_boost_round=parameters["n_estimators"],
        evals=[(dtrain, "train")],
        early_stopping_rounds=30,
        verbose_eval=False
    )

    # Predicciones y evaluación en test
    probs_test = booster.predict(dtest)
    best_thr = parameters.get("threshold", 0.5)
    preds_test = (probs_test > best_thr).astype(int)

    report = classification_report(y_test, preds_test)

    print("→ Reporte final en TEST:")
    print(report)

    # Guardar modelo
    joblib.dump(booster, os.path.join(execution_date, "train/xgb_model.bin"))

    # Guardar threshold
    with open(os.path.join(execution_date, "train/threshold.txt", "w")) as f:
        f.write(str(best_thr))

    # Guardar reporte
    with open(os.path.join(execution_date, "train/classification_report.txt", "w")) as f:
        f.write(report)


def export_model(model, path):
    pass