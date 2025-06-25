"""
estructura planteada:

run_prediction: Ejecuta la predicción en los datos de entrada
get_products: Obtiene las predicciones del modelo de la clase positiva (par cliente-producto que si compra)

"""

import pandas as pd
import joblib
import xgboost as xgb
import os
import numpy as np
from datetime import datetime, timedelta
import itertools


def create_robust_prediction_dataset(df_processed, next_week, max_week, client_stats, weekly_stats, product_stats):
    """
    Crea un dataset robusto para predicciones, manejando casos edge y optimizando rendimiento.
    Usa solo estadísticas de entrenamiento.
    """
    print(" Construyendo dataset de predicción...")
    
    # Obtener datos históricos solo hasta la semana actual
    historical_data = df_processed[df_processed['week'] <= max_week].copy()
    
    # Obtener combinaciones únicas de cliente-producto que han existido
    unique_customers = historical_data['customer_id'].unique()
    unique_products = historical_data['product_id'].unique()
    
    print(f"Generando predicciones para {len(unique_customers)} clientes x {len(unique_products)} productos")
    
    # Crear todas las combinaciones posibles
    combinations = list(itertools.product(unique_customers, unique_products))
    
    # Crear DataFrame base con todas las combinaciones
    prediction_df = pd.DataFrame(combinations, columns=['customer_id', 'product_id'])
    prediction_df['week'] = next_week
    prediction_df['purchase_date'] = pd.NaT
    prediction_df['label'] = 0  # Lo que queremos predecir
    
    print(f"Combinaciones totales: {len(prediction_df):,}")
    
    # PASO 1: Agregar features temporales de manera vectorizada
    print("Calculando features temporales...")
    
    # Obtener último registro de cada par cliente-producto
    last_records = (historical_data
                    .sort_values('week')
                    .groupby(['customer_id', 'product_id'])
                    .last()
                    .reset_index())
    
    # Calcular items_last_week (usar items_last_week existente o 0 como default)
    items_last_week = last_records[['customer_id', 'product_id', 'items_last_week']]
    
    # Calcular rolling mean de las últimas 4 semanas (usar items_last_week como proxy)
    rolling_means = []
    for (customer_id, product_id), group in historical_data.groupby(['customer_id', 'product_id']):
        if len(group) >= 4:
            mean_4w = group.sort_values('week')['items_last_week'].tail(4).mean()
        elif len(group) > 0:
            mean_4w = group['items_last_week'].mean()
        else:
            mean_4w = 0
        
        rolling_means.append({
            'customer_id': customer_id,
            'product_id': product_id,
            'items_roll4_mean': mean_4w
        })
    
    items_roll4_mean = pd.DataFrame(rolling_means)
    
    # PASO 2: Merge con features temporales
    prediction_df = prediction_df.merge(
        items_last_week, on=['customer_id', 'product_id'], how='left'
    ).fillna({'items_last_week': 0})
    
    prediction_df = prediction_df.merge(
        items_roll4_mean, on=['customer_id', 'product_id'], how='left'
    ).fillna({'items_roll4_mean': 0})
    
    # PASO 3: Agregar información de cliente y producto
    print("Agregando información de clientes y productos...")
    
    # Obtener info única de cada cliente y producto
    client_info = historical_data[['customer_id', 'customer_type', 'Y', 'X', 'num_deliver_per_week', 'custom_zone']].drop_duplicates()
    product_info = historical_data[['product_id', 'brand', 'category', 'sub_category', 'segment', 'package', 'size']].drop_duplicates()
    
    # Merge con información estática
    prediction_df = prediction_df.merge(client_info, on='customer_id', how='left')
    prediction_df = prediction_df.merge(product_info, on='product_id', how='left')
    
    # PASO 4: Manejo de casos edge (clientes/productos nuevos)
    # Si hay valores faltantes, usar valores por defecto
    # Para columnas categóricas, usar el valor más frecuente en lugar de valores nuevos
    
    # Llenar valores faltantes de manera segura
    if 'customer_type' in prediction_df.columns:
        if prediction_df['customer_type'].dtype.name == 'category':
            # Para categóricas, usar el modo (valor más frecuente)
            mode_val = prediction_df['customer_type'].mode()
            if len(mode_val) > 0:
                prediction_df['customer_type'] = prediction_df['customer_type'].fillna(mode_val[0])
        else:
            prediction_df['customer_type'] = prediction_df['customer_type'].fillna('ABARROTES')
    
    # Para columnas numéricas, usar valores por defecto seguros
    numeric_defaults = {
        'Y': prediction_df['Y'].median() if 'Y' in prediction_df.columns else 0.0,
        'X': prediction_df['X'].median() if 'X' in prediction_df.columns else 0.0,
        'num_deliver_per_week': prediction_df['num_deliver_per_week'].mode()[0] if 'num_deliver_per_week' in prediction_df.columns else 1,
        'custom_zone': prediction_df['custom_zone'].mode()[0] if 'custom_zone' in prediction_df.columns else 0,
        'size': prediction_df['size'].median() if 'size' in prediction_df.columns else 500.0
    }
    
    for col, default_val in numeric_defaults.items():
        if col in prediction_df.columns:
            prediction_df[col] = prediction_df[col].fillna(default_val)
    
    # Para columnas categóricas de productos, usar modo
    categorical_product_cols = ['brand', 'category', 'sub_category', 'segment', 'package']
    for col in categorical_product_cols:
        if col in prediction_df.columns:
            if prediction_df[col].dtype.name == 'category':
                mode_val = prediction_df[col].mode()
                if len(mode_val) > 0:
                    prediction_df[col] = prediction_df[col].fillna(mode_val[0])
            else:
                # Si no es categórica, usar valores por defecto del primer producto válido
                first_valid = prediction_df[col].dropna().iloc[0] if len(prediction_df[col].dropna()) > 0 else 'Unknown'
                prediction_df[col] = prediction_df[col].fillna(first_valid)
    
    print(f"Dataset de predicción creado: {len(prediction_df):,} registros")
    print(f"   - Features temporales calculadas")
    print(f"   - Información de clientes y productos agregada")
    print(f"   - Casos edge manejados con valores por defecto")
    
    return prediction_df


def run_prediction(**kwargs):
    """
    Ejecuta predicciones usando el modelo exportado.
    Genera predicciones para la próxima semana (semana siguiente a la más reciente en los datos).
    Sin data leakage y robusta para producción.
    """
    execution_date = kwargs['ds']
    
    print("Iniciando generación de predicciones...")
    
    # Cargar modelo exportado y artefactos
    model_dir = os.path.join(execution_date, "model_export")
    
    # Cargar modelo, pipeline y threshold
    model = joblib.load(os.path.join(model_dir, "model.bin"))
    features_pipeline = joblib.load(os.path.join(model_dir, "features_pipeline.pkl"))
    
    with open(os.path.join(model_dir, "threshold.txt"), "r") as f:
        threshold = float(f.read().strip())
    
    # Cargar estadísticas desde el modelo exportado
    stats_dir = os.path.join(model_dir, "feature_stats")
    client_stats = pd.read_parquet(os.path.join(stats_dir, "client_stats.parquet"))
    weekly_stats = pd.read_parquet(os.path.join(stats_dir, "weekly_stats.parquet"))
    product_stats = pd.read_parquet(os.path.join(stats_dir, "product_stats.parquet"))
    
    print(f"Estadísticas cargadas:")
    print(f"   - Clientes: {len(client_stats)}")
    print(f"   - Promedios semanales: {len(weekly_stats)}")
    print(f"   - Productos: {len(product_stats)}")
    
    # Cargar datos procesados originales para obtener información necesaria
    df_processed = pd.read_parquet(os.path.join(execution_date, "data_processed/df_processed.parquet"))
    
    # Obtener la semana más reciente en los datos
    max_week = df_processed['week'].max()
    next_week = max_week + 1
    
    print(f"Semana más reciente en datos: {max_week}")
    print(f"Generando predicciones para semana: {next_week}")
    
    # Crear dataset de predicción de manera eficiente
    prediction_df = create_robust_prediction_dataset(
        df_processed, next_week, max_week, client_stats, weekly_stats, product_stats
    )
    
    if len(prediction_df) == 0:
        print("No se pudieron generar datos para predicción")
        return
    
    print(f"Datos preparados para predicción: {len(prediction_df)} registros")
    
    # Separar features 
    X_pred = prediction_df.drop(columns=['label'])
    
    # Aplicar el pipeline de features
    try:
        X_pred_transformed = features_pipeline.transform(X_pred)
        print(f"Features transformadas: {X_pred_transformed.shape}")
    except Exception as e:
        print(f"Error en transformación de features: {e}")
        return
    
    # Hacer predicciones
    dmatrix_pred = xgb.DMatrix(X_pred_transformed)
    probabilities = model.predict(dmatrix_pred)
    predictions = (probabilities > threshold).astype(int)
    
    # Agregar predicciones al DataFrame
    prediction_df['probability'] = probabilities
    prediction_df['prediction'] = predictions
    
    print(f"Predicciones generadas:")
    print(f"   - Total: {len(predictions)}")
    print(f"   - Positivas: {predictions.sum()}")
    print(f"   - Negativas: {len(predictions) - predictions.sum()}")
    print(f"   - Threshold usado: {threshold:.4f}")
    
    # Guardar resultados
    predictions_dir = os.path.join(execution_date, "predictions")
    os.makedirs(predictions_dir, exist_ok=True)
    
    prediction_df.to_parquet(
        os.path.join(predictions_dir, "all_predictions.parquet"), 
        index=False
    )
    
    print(f"Predicciones guardadas en: {predictions_dir}/all_predictions.parquet")
    

def get_products(**kwargs):
    """
    Obtiene las predicciones positivas y genera el archivo CSV final.
    Solo incluye las combinaciones cliente-producto con predicción positiva.
    """
    execution_date = kwargs['ds']
    
    print("Generando archivo CSV con productos recomendados...")
    
    # Cargar predicciones
    predictions_dir = os.path.join(execution_date, "predictions")
    all_predictions = pd.read_parquet(os.path.join(predictions_dir, "all_predictions.parquet"))
    
    # Filtrar solo predicciones positivas
    positive_predictions = all_predictions[all_predictions['prediction'] == 1].copy()
    
    print(f"Predicciones positivas encontradas: {len(positive_predictions)}")
    
    if len(positive_predictions) == 0:
        print("No se encontraron predicciones positivas")
        # Crear archivo vacío
        empty_df = pd.DataFrame(columns=['customer_id', 'product_id', 'week', 'probability'])
        empty_df.to_csv(os.path.join(predictions_dir, "recommended_products.csv"), index=False)
        return
    
    # Seleccionar solo las columnas necesarias para el archivo final
    final_recommendations = positive_predictions[[
        'customer_id', 
        'product_id', 
        'week', 
        'probability'
    ]].copy()
    
    # Ordenar por probabilidad descendente
    final_recommendations = final_recommendations.sort_values('probability', ascending=False)
    
    # Generar archivo CSV
    csv_path = os.path.join(predictions_dir, "recommended_products.csv")
    final_recommendations.to_csv(csv_path, index=False)
    
    print(f"Resumen de recomendaciones:")
    print(f"   - Total de recomendaciones: {len(final_recommendations)}")
    print(f"   - Clientes únicos: {final_recommendations['customer_id'].nunique()}")
    print(f"   - Productos únicos: {final_recommendations['product_id'].nunique()}")
    print(f"   - Semana objetivo: {final_recommendations['week'].iloc[0]}")
    print(f"   - Probabilidad promedio: {final_recommendations['probability'].mean():.4f}")
    print(f"   - Probabilidad máxima: {final_recommendations['probability'].max():.4f}")
    print(f"   - Probabilidad mínima: {final_recommendations['probability'].min():.4f}")
    
    print(f"Archivo CSV generado: {csv_path}")
    
    # Mostrar ejemplos de recomendaciones
    print(f"\nPrimeras 10 recomendaciones:")
    print(final_recommendations.head(10).to_string(index=False))
    
    return csv_path
