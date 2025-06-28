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
import joblib
import json
import os

SHARED_DATA_DIR = "/shared-data"  

def detect_drift(**kwargs):
    """
    Detecta drift en los datos comparando distribuciones de features con referencias históricas.
    Maneja casos borde como primera ejecución.
    """
    execution_date = kwargs['ds']
    base_path = os.path.join(SHARED_DATA_DIR, execution_date)
    
    print(" Detectando drift en los datos...")
    
    try:
        from scipy.stats import ks_2samp
        
        # Cargar datos actuales
        current_data = pd.read_parquet(os.path.join(base_path, "data_processed/df_processed.parquet"))
        
        # Buscar datos de referencia
        reference_data = None
        reference_source = "unknown"
        
        # 1. Intentar cargar desde ejecución anterior 
        try:
            # Buscar la carpeta de fecha anterior más reciente con modelo
            import glob
            from datetime import datetime, timedelta
            
            current_date = datetime.strptime(execution_date, "%Y-%m-%d")
            for days_back in range(7, 365, 7):  # Buscar semanas anteriores
                prev_date = current_date - timedelta(days=days_back)
                prev_date_str = prev_date.strftime("%Y-%m-%d")
                prev_model_path = os.path.join(prev_date_str, "model_export/model.bin")
                prev_data_path = os.path.join(prev_date_str, "data_processed/df_processed.parquet")
                
                if os.path.exists(prev_model_path) and os.path.exists(prev_data_path):
                    reference_data = pd.read_parquet(prev_data_path)
                    reference_source = f"modelo_previo_{prev_date_str}"
                    print(f"📊 Usando datos de referencia desde: {prev_date_str}")
                    break
        except Exception as e:
            print(f" No se encontraron datos de modelo previo: {e}")
        
        # 2. Si no hay modelo previo, usar datos históricos 
        if reference_data is None:
            try:
                reference_data = pd.read_parquet(os.path.join(base_path, "data_holdout/train_df.parquet"))
                reference_source = "baseline_historico"
                print(f" Usando datos baseline históricos como referencia")
            except Exception as e:
                print(f" No se encontraron datos de referencia: {e}")
                return False
        
        # Obtener muestra representativa de datos actuales (últimas 2 semanas)
        max_week = current_data['week'].max()
        recent_data = current_data[current_data['week'] >= max_week - 1]
        
        # Features para análisis de drift 
        numeric_features = [
            'Y', 'X', 'num_deliver_per_week', 'size', 
            'items_last_week', 'items_roll4_mean'
        ]
        
        categorical_features = ['category', 'brand'] if 'category' in current_data.columns else []
        
        drift_results = {}
        drift_detected = False
        drift_score = 0.0
        p_value_threshold = 0.05
        
        # Análisis de drift en features numéricas
        for feature in numeric_features:
            if feature in reference_data.columns and feature in recent_data.columns:
                ref_values = reference_data[feature].dropna()
                new_values = recent_data[feature].dropna()
                
                if len(ref_values) > 30 and len(new_values) > 30:  
                    statistic, p_value = ks_2samp(ref_values, new_values)
                    
                    # Calcular diferencia en medias y varianzas para contexto
                    mean_diff = abs(new_values.mean() - ref_values.mean()) / (ref_values.std() + 1e-8)
                    var_ratio = new_values.var() / (ref_values.var() + 1e-8)
                    
                    feature_drift = p_value < p_value_threshold
                    if feature_drift:
                        drift_score += 1.0
                    
                    drift_results[feature] = {
                        'p_value': float(p_value),
                        'statistic': float(statistic),
                        'mean_diff_normalized': float(mean_diff),
                        'variance_ratio': float(var_ratio),
                        'drift_detected': bool(feature_drift),
                        'reference_source': str(reference_source)
                    }
                    
                    status = " DRIFT" if feature_drift else " OK"
                    print(f"   {status} {feature}: p={p_value:.4f}, mean_diff={mean_diff:.3f}")
        
        # Análisis de drift en features categóricas (distribuciones)
        for feature in categorical_features:
            if feature in reference_data.columns and feature in recent_data.columns:
                try:
                    ref_dist = reference_data[feature].value_counts(normalize=True)
                    new_dist = recent_data[feature].value_counts(normalize=True)
                    
                    # Chi-cuadrado para distribuciones categóricas
                    from scipy.stats import chisquare
                    
                    common_categories = set(ref_dist.index) & set(new_dist.index)
                    if len(common_categories) > 1:
                        ref_probs = [ref_dist.get(cat, 0) for cat in common_categories]
                        new_probs = [new_dist.get(cat, 0) for cat in common_categories]
                        
                        if sum(new_probs) > 0:
                            statistic, p_value = chisquare(new_probs, ref_probs)
                            feature_drift = p_value < p_value_threshold
                            
                            if feature_drift:
                                drift_score += 0.5  # Menos peso que variables numéricas
                            
                            drift_results[f"{feature}_categorical"] = {
                                'p_value': float(p_value),
                                'statistic': float(statistic),
                                'drift_detected': bool(feature_drift),
                                'reference_source': str(reference_source)
                            }
                            
                            status = " DRIFT" if feature_drift else " OK"
                            print(f"   {status} {feature} (categorical): p={p_value:.4f}")
                except Exception as e:
                    print(f" Error analizando feature categórica {feature}: {e}")
        
        # Decisión final de drift
        drift_threshold = 2.0  # Umbral para considerar drift significativo
        drift_detected = drift_score >= drift_threshold
        
        # Metadatos del análisis 
        analysis_metadata = {
            'total_drift_score': float(drift_score),
            'drift_threshold': float(drift_threshold),
            'drift_detected': bool(drift_detected),
            'reference_source': str(reference_source),
            'features_analyzed': int(len(drift_results)),
            'current_data_size': int(len(recent_data)),
            'reference_data_size': int(len(reference_data)),
            'execution_date': str(execution_date)
        }
        
        # Guardar resultados completos
        drift_dir = os.path.join(base_path, "drift_detection")
        os.makedirs(drift_dir, exist_ok=True)
        
        complete_results = {
            'metadata': analysis_metadata,
            'feature_analysis': drift_results
        }
        
        # Guardar con manejo robusto de errores
        try:
            with open(os.path.join(drift_dir, "drift_results.json"), "w") as f:
                json.dump(complete_results, f, indent=2, default=str)  # default=str para manejar objetos no serializables
            print(f" Resultados de drift guardados exitosamente")
        except Exception as e:
            print(f" Error guardando resultados detallados: {e}")
            # Guardar versión simplificada como fallback
            simplified_results = {
                'drift_detected': bool(drift_detected),
                'drift_score': float(drift_score),
                'reference_source': str(reference_source),
                'execution_date': str(execution_date)
            }
            with open(os.path.join(drift_dir, "drift_results.json"), "w") as f:
                json.dump(simplified_results, f, indent=2)
        
        # Guardar decisión simple para branching
        with open(os.path.join(drift_dir, "drift_detected.txt"), "w") as f:
            f.write("true" if drift_detected else "false")
        
        # Logs finales
        if drift_detected:
            print(f" DRIFT DETECTADO - Score: {drift_score:.1f}/{drift_threshold}")
            print(f"   Fuente referencia: {reference_source}")
            print(f"   Reentrenamiento RECOMENDADO")
        else:
            print(f" Sin drift significativo - Score: {drift_score:.1f}/{drift_threshold}")
            print(f"   Fuente referencia: {reference_source}")
            print(f"   Modelo actual sigue siendo válido")
            
        return drift_detected
        
    except ImportError:
        print(" scipy no disponible, asumiendo drift para forzar reentrenamiento")
        return True
    except Exception as e:
        print(f" Error en detección de drift: {e}")
        print(f"   Asumiendo drift por seguridad - forzando reentrenamiento")
        return True


def decide_retraining(**kwargs):
    """
    Decide si reentrenar basado en drift detection y casos borde.
    Retorna task_id para branching.
    """
    execution_date = kwargs['ds']
    base_path = os.path.join(SHARED_DATA_DIR, execution_date)
    
    print(" Decidiendo estrategia de reentrenamiento...")
    print(f"   Execution date: {execution_date}")
    
    # Verificar si existe modelo previo
    model_export_path = os.path.join(base_path, "model_export")
    has_previous_model = False
    
    print(f"   Buscando modelos previos...")
    
    # Buscar modelo en fechas anteriores
    try:
        from datetime import datetime, timedelta
        
        current_date = datetime.strptime(execution_date, "%Y-%m-%d")
        print(f"   Fecha actual parseada: {current_date}")
        
        for days_back in range(7, 365, 7):  # Buscar semanas anteriores
            prev_date = current_date - timedelta(days=days_back)
            prev_date_str = prev_date.strftime("%Y-%m-%d")
            prev_model_path = os.path.join(prev_date_str, "model_export/model.bin")
            
            print(f"   Verificando: {prev_model_path}")
            
            if os.path.exists(prev_model_path):
                has_previous_model = True
                print(f" Modelo previo encontrado en: {prev_date_str}")
                break
        
        if not has_previous_model:
            print(f"   No se encontraron modelos previos en 365 días")
            
    except Exception as e:
        print(f" Error buscando modelo previo: {e}")
        import traceback
        print(f"   Traceback: {traceback.format_exc()}")
    
    # Caso 1: Primera ejecución - SIEMPRE entrenar
    if not has_previous_model:
        print(" PRIMERA EJECUCIÓN - Entrenando modelo inicial")
        return 'optimize_hyperparameters'
    
    # Caso 2: Leer resultado de drift detection
    print(f"   Leyendo resultado de drift detection...")
    try:
        drift_file = os.path.join(base_path, "drift_detection/drift_detected.txt")
        print(f"   Archivo drift: {drift_file}")
        
        if not os.path.exists(drift_file):
            print(f" Archivo drift no existe, reentrenando por seguridad")
            return 'optimize_hyperparameters'
        
        with open(drift_file, "r") as f:
            drift_content = f.read().strip()
            drift_detected = drift_content.lower() == "true"
        
        print(f"   Contenido drift file: '{drift_content}'")
        print(f"   Drift detected: {drift_detected}")
        
        if drift_detected:
            print(" DRIFT DETECTADO - Reentrenando modelo")
            return 'optimize_hyperparameters'
        else:
            print(" SIN DRIFT - Reutilizando modelo previo")
            return 'copy_previous_model'
            
    except Exception as e:
        print(f" Error leyendo drift detection: {e}")
        import traceback
        print(f"   Traceback: {traceback.format_exc()}")
        print(" Reentrenando por seguridad")
        return 'optimize_hyperparameters'   


def copy_previous_model(**kwargs):
    """
    Copia el modelo más reciente disponible cuando no hay drift.
    """
    execution_date = kwargs['ds']
    base_path = os.path.join(SHARED_DATA_DIR, execution_date)
    
    print(" Copiando modelo previo (sin drift detectado)...")
    
    try:
        import shutil
        from datetime import datetime, timedelta
        
        current_date = datetime.strptime(execution_date, "%Y-%m-%d")
        
        # Buscar modelo más reciente
        for days_back in range(7, 365, 7):
            prev_date = current_date - timedelta(days=days_back)
            prev_date_str = prev_date.strftime("%Y-%m-%d")
            prev_export_dir = os.path.join(prev_date_str, "model_export")
            
            if os.path.exists(prev_export_dir):
                current_export_dir = os.path.join(base_path, "model_export")
                
                # Crear directorio destino
                os.makedirs(current_export_dir, exist_ok=True)
                
                # Copiar todos los artefactos
                for item in os.listdir(prev_export_dir):
                    src = os.path.join(prev_export_dir, item)
                    dst = os.path.join(current_export_dir, item)
                    
                    if os.path.isfile(src):
                        shutil.copy2(src, dst)
                    elif os.path.isdir(src):
                        if os.path.exists(dst):
                            shutil.rmtree(dst)
                        shutil.copytree(src, dst)
                
                print(f" Modelo copiado desde: {prev_date_str}")
                
                # Crear metadatos de copia
                copy_metadata = {
                    'copied_from': prev_date_str,
                    'copy_date': execution_date,
                    'reason': 'no_drift_detected'
                }
                
                with open(os.path.join(current_export_dir, "copy_metadata.json"), "w") as f:
                    json.dump(copy_metadata, f, indent=2)
                
                return
        
        raise Exception("No se encontró modelo previo para copiar")
        
    except Exception as e:
        print(f" Error copiando modelo previo: {e}")
        print(f" Fallback: forzando reentrenamiento")
        raise

# TO-DO: agregar mlflow para trackear los resultados
def optimize_hyperparameters(**kwargs):

    execution_date = kwargs['ds']
    base_path = os.path.join(SHARED_DATA_DIR, execution_date)
    # Load the data
    X_train_tr = pd.read_parquet(os.path.join(base_path, "data_transformed/X_train.parquet"))
    y_train = pd.read_parquet(os.path.join(base_path, "data_transformed/y_train.parquet"))["label"]

    X_val_tr = pd.read_parquet(os.path.join(base_path, "data_transformed/X_val.parquet"))
    y_val = pd.read_parquet(os.path.join(base_path, "data_transformed/y_val.parquet"))["label"]

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

    # Agregar threshold óptimo a los parámetros
    params_final["threshold"] = best_thr
    params_final["best_f1_validation"] = best_f1
    
    print(f"Hiperparámetros óptimos encontrados:")
    print(f"   - Threshold óptimo: {best_thr:.4f}")
    print(f"   - F1-Score en validación: {best_f1:.4f}")
    print(f"   - N estimators: {n_rounds_final}")
    print(f"   - Learning rate: {params_final['learning_rate']:.4f}")
    print(f"   - Max depth: {params_final['max_depth']}")
    print(f"   - Subsample: {params_final['subsample']:.4f}")
    
  
    print(f" Preprocessing fijo usado: SimpleImputer(median) + MinMaxScaler + OneHotEncoder")

    # guardar los resultados
    os.makedirs(os.path.join(base_path, "train"), exist_ok=True)
    with open(os.path.join(base_path, "train/optuna_params.json"), "w") as f:
        json.dump(params_final, f, indent=2)


def train_model(**kwargs):
    
    execution_date = kwargs['ds']
    base_path = os.path.join(SHARED_DATA_DIR, execution_date)
    # Cargar datos
    X_train_tr = pd.read_parquet(os.path.join(base_path, "data_transformed/X_train.parquet"))
    y_train = pd.read_parquet(os.path.join(base_path, "data_transformed/y_train.parquet"))["label"]

    X_test = pd.read_parquet(os.path.join(base_path, "data_transformed/X_test.parquet"))
    y_test = pd.read_parquet(os.path.join(base_path, "data_transformed/y_test.parquet"))["label"]

    dtrain = xgb.DMatrix(X_train_tr, label=y_train)
    dtest = xgb.DMatrix(X_test)

    # Cargar parámetros de Optuna
    with open(os.path.join(base_path, "train/optuna_params.json"), "r") as f:
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
    joblib.dump(booster, os.path.join(base_path, "train/xgb_model.bin"))

    # Guardar threshold
    with open(os.path.join(base_path, "train/threshold.txt"), "w") as f:
        f.write(str(best_thr))

    # Guardar reporte
    with open(os.path.join(base_path, "train/classification_report.txt"), "w") as f:
        f.write(report)


def evaluate_model(**kwargs):
    """
    Evalúa el modelo entrenado y registra métricas con MLflow (opcional).
    """
    execution_date = kwargs['ds']
    base_path = os.path.join(SHARED_DATA_DIR, execution_date)
    
    # Cargar modelo entrenado
    model = joblib.load(os.path.join(base_path, "train/xgb_model.bin"))
    
    # Cargar datos de test
    X_test = pd.read_parquet(os.path.join(base_path, "data_transformed/X_test.parquet"))
    y_test = pd.read_parquet(os.path.join(base_path, "data_transformed/y_test.parquet"))["label"]
    
    # Cargar threshold
    with open(os.path.join(base_path, "train/threshold.txt"), "r") as f:
        threshold = float(f.read().strip())
    
    # Predicciones
    dtest = xgb.DMatrix(X_test)
    probs_test = model.predict(dtest)
    preds_test = (probs_test > threshold).astype(int)
    
    # Métricas
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    
    metrics = {
        'accuracy': accuracy_score(y_test, preds_test),
        'precision': precision_score(y_test, preds_test),
        'recall': recall_score(y_test, preds_test),
        'f1_score': f1_score(y_test, preds_test),
        'roc_auc': roc_auc_score(y_test, probs_test),
        'threshold': threshold
    }
    
    print("Métricas del modelo:")
    for metric, value in metrics.items():
        print(f"   - {metric}: {value:.4f}")
    
    # Guardar métricas
    with open(os.path.join(base_path, "train/metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    
    return metrics


def export_model(**kwargs):
    """
    Exporta el modelo entrenado y sus artefactos.
    """
    execution_date = kwargs['ds']
    base_path = os.path.join(SHARED_DATA_DIR, execution_date)
    
    # Crear directorio de modelos exportados
    export_dir = os.path.join(base_path, "model_export")
    os.makedirs(export_dir, exist_ok=True)
    
    # Copiar modelo entrenado
    import shutil
    shutil.copy2(
        os.path.join(base_path, "train/xgb_model.bin"),
        os.path.join(export_dir, "model.bin")
    )
    
    # Copiar pipeline de features
    shutil.copy2(
        os.path.join(base_path, "data_transformed/features_pipeline.pkl"),
        os.path.join(export_dir, "features_pipeline.pkl")
    )
    
    # Copiar threshold
    shutil.copy2(
        os.path.join(base_path, "train/threshold.txt"),
        os.path.join(export_dir, "threshold.txt")
    )
    
    # Copiar métricas
    shutil.copy2(
        os.path.join(base_path, "train/metrics.json"),
        os.path.join(export_dir, "metrics.json")
    )
    
    # ✅ NUEVO: Copiar estadísticas de features para predicción
    stats_source = os.path.join(base_path, "feature_stats")
    stats_dest = os.path.join(export_dir, "feature_stats")
    
    if os.path.exists(stats_source):
        if os.path.exists(stats_dest):
            shutil.rmtree(stats_dest)  # Eliminar directorio existente primero
        shutil.copytree(stats_source, stats_dest)
        print(f"Estadísticas de features copiadas a modelo exportado")
    else:
        print(f"No se encontraron estadísticas de features en {stats_source}")
    
    print(f" Modelo exportado exitosamente a: {export_dir}")
    return export_dir