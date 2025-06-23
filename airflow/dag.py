"""
Base para el DAG de Airflow
"""

from airflow import DAG
from airflow.operators.empty import EmptyOperator
from airflow.operators.python import PythonOperator, BranchPythonOperator
import pandas as pd
from datetime import datetime
from data_functions import get_data, process_data, holdout, feature_engineering
from train_functions import detect_drift, optimize_hyperparameters, train_model, evaluate_model, export_model
from predictions_functions import run_prediction, get_products


default_args = {
    'start_date': datetime(2024, 10, 1),
}

with DAG(
    dag_id='sodAI',
    default_args=default_args,
    schedule='0 15 5 * *',   # cambiar
    catchup=True,            # habilita backfill
    tags=['sodAI']
) as dag:

    # 1) Placeholder de inicio
    start = EmptyOperator(task_id='start')

    # 2) Obtener datos
    get_data = PythonOperator(
        task_id='create_folders',
        python_callable=get_data
    )

    # 3) Procesar datos
    process_data = PythonOperator(
        task_id='process_data',
        python_callable=process_data
    )

    # 4) Holdout
    holdout = BranchPythonOperator(
        task_id='holdout',
        python_callable=holdout,
    )

    # 5) Feature engineering
    feature_engineering = PythonOperator(
        task_id='feature_engineering',
        python_callable=feature_engineering
    )

    # 6) Detectar drift
    detect_drift = PythonOperator(
        task_id='detect_drift',
        python_callable=detect_drift
    )

    # 7) Optimizar hiperparámetros
    optimize_hyperparameters = PythonOperator(
        task_id='optimize_hyperparameters',
        python_callable=optimize_hyperparameters
    )

    # 8) Entrenar modelo
    train_model = PythonOperator(
        task_id='train_model',
        python_callable=train_model
    )

    # 9) Evaluar modelo
    evaluate_model = PythonOperator(
        task_id='evaluate_model',
        python_callable=evaluate_model
    )

    # 10) Exportar modelo
    export_model = PythonOperator(
        task_id='export_model',
        python_callable=export_model
    )

    # 11) Ejecutar predicciones
    run_prediction = PythonOperator(
        task_id='run_prediction',
        python_callable=run_prediction
    )

    # 12) Obtener productos
    get_products = PythonOperator(
        task_id='get_products',
        python_callable=get_products
    )

    # 13) Placeholder de finalización
    end = EmptyOperator(task_id='end')

    # Definición del flujo
    (
        start 
        >> get_data 
        >> process_data 
        >> feature_engineering 
        >> export_data 
        >> detect_drift 
        >> optimize_hyperparameters 
        >> train_model 
        >> evaluate_model 
        >> export_model 
        >> run_prediction 
        >> get_products 
        >> end
    )

    
