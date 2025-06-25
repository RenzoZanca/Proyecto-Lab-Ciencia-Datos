"""
Base para el DAG de Airflow
"""

from airflow import DAG
from airflow.operators.empty import EmptyOperator
from airflow.operators.python import PythonOperator, BranchPythonOperator
import pandas as pd
from datetime import datetime
import os
from data_functions import get_data, process_data, holdout, feature_engineering
from train_functions import detect_drift, decide_retraining, copy_previous_model, optimize_hyperparameters, train_model, evaluate_model, export_model
from predictions_functions import run_prediction, get_products


def check_data_exists(**kwargs):
    """
    Verifica si existen datos para la fecha de ejecución.
    Si no existen, salta el procesamiento .
    """
    execution_date = kwargs['ds']
    data_path = os.path.join(execution_date, "data")
    
    if os.path.exists(data_path):
        files_needed = ['transacciones.parquet', 'clientes.parquet', 'productos.parquet']
        if all(os.path.exists(os.path.join(data_path, f)) for f in files_needed):
            print(f" Datos encontrados para {execution_date}. Continuando pipeline...")
            return 'get_data'
        else:
            print(f" Datos incompletos para {execution_date}. Saltando ejecución...")
            return 'skip_processing'
    else:
        print(f" No hay datos para {execution_date}. ")
        return 'skip_processing'


default_args = {
    'start_date': datetime(2024, 10, 1),
}

with DAG(
    dag_id='sodAI',
    default_args=default_args,
    schedule='@weekly',  # Ejecutar semanalmente
    catchup=True,        # HABILITADO para simular comportamiento productivo
    max_active_runs=1,   # Evita ejecuciones paralelas que saturen recursos
    tags=['sodAI']
) as dag:

    # 1) Verificación de datos 
    check_data = BranchPythonOperator(
        task_id='check_data',
        python_callable=check_data_exists
    )

    # 2) Skip processing 
    skip_processing = EmptyOperator(
        task_id='skip_processing'
    )

    # 3) Placeholder de inicio
    start = EmptyOperator(task_id='start')

    # 4) Obtener datos
    get_data_task = PythonOperator(
        task_id='get_data',
        python_callable=get_data
    )

    # 5) Procesar datos
    process_data_task = PythonOperator(
        task_id='process_data',
        python_callable=process_data
    )

    # 6) Holdout
    holdout_task = PythonOperator(
        task_id='holdout',
        python_callable=holdout,
    )

    # 7) Feature engineering
    feature_engineering_task = PythonOperator(
        task_id='feature_engineering',
        python_callable=feature_engineering
    )

    # 8) Detectar drift
    detect_drift_task = PythonOperator(
        task_id='detect_drift',
        python_callable=detect_drift
    )

    # 9) Decisión de reentrenamiento (branching)
    decide_retraining_task = BranchPythonOperator(
        task_id='decide_retraining',
        python_callable=decide_retraining
    )

    # 10a) Copiar modelo previo (sin drift)
    copy_previous_model_task = PythonOperator(
        task_id='copy_previous_model',
        python_callable=copy_previous_model
    )

    # 10b) Optimizar hiperparámetros (con drift o primera vez)
    optimize_hyperparameters_task = PythonOperator(
        task_id='optimize_hyperparameters',
        python_callable=optimize_hyperparameters
    )

    # 11) Entrenar modelo
    train_model_task = PythonOperator(
        task_id='train_model',
        python_callable=train_model
    )

    # 12) Evaluar modelo
    evaluate_model_task = PythonOperator(
        task_id='evaluate_model',
        python_callable=evaluate_model
    )

    # 13) Exportar modelo
    export_model_task = PythonOperator(
        task_id='export_model',
        python_callable=export_model
    )

    # 14) Unión de flujos (dummy task)
    model_ready = EmptyOperator(
        task_id='model_ready',
        trigger_rule='none_failed_min_one_success'  # Funciona si cualquiera de los dos flujos anteriores tuvo éxito
    )

    # 15) Ejecutar predicciones
    run_prediction_task = PythonOperator(
        task_id='run_prediction',
        python_callable=run_prediction
    )

    # 16) Obtener productos
    get_products_task = PythonOperator(
        task_id='get_products',
        python_callable=get_products
    )

    # 17) Placeholder de finalización
    end = EmptyOperator(task_id='end')

    # Definición del flujo con lógica condicional completa
    
    # 1. Flujo inicial con verificación de datos
    (
        start 
        >> check_data
    )
    
    # 2. Flujo principal de procesamiento si hay datos
    (
        check_data
        >> get_data_task
        >> process_data_task 
        >> holdout_task
        >> feature_engineering_task 
        >> detect_drift_task 
        >> decide_retraining_task
    )
    
    # 3a. Flujo de reentrenamiento 
    (
        decide_retraining_task
        >> optimize_hyperparameters_task 
        >> train_model_task 
        >> evaluate_model_task 
        >> export_model_task 
        >> model_ready
    )
    
    # 3b. Flujo de copia de modelo (sin drift)
    (
        decide_retraining_task
        >> copy_previous_model_task
        >> model_ready
    )
    
    # 4. Flujo de predicciones (unión de ambos flujos anteriores)
    (
        model_ready
        >> run_prediction_task 
        >> get_products_task 
        >> end
    )
    
    # 5. Flujo de skip si no hay datos
    (
        check_data
        >> skip_processing
        >> end
    )

    
