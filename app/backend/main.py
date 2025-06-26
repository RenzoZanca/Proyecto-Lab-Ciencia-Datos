from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
import os
from datetime import datetime
import shutil
import subprocess
import requests
import time

app = FastAPI()

def wait_for_dag_status(dag_id: str, execution_date: str, timeout=2400, interval=60):
    """
    Espera hasta que el DAG finalice (success o failed).
    - execution_date debe estar en formato ISO 8601: YYYY-MM-DDTHH:MM:SS+00:00
    """
    base_url = "http://localhost:8080/api/v1"
    elapsed = 0

    while elapsed < timeout:
        url = f"{base_url}/dags/{dag_id}/dagRuns/{execution_date}"
        response = requests.get(url)

        if response.status_code == 200:
            state = response.json()["state"]
            print(f"[{elapsed}s] DAG state: {state}")
            if state == "success":
                return "success"
            elif state in ["failed", "error"]:
                return "failed"
        else:
            print(f"⚠️ Error fetching DAG status: {response.status_code}")

        time.sleep(interval)
        elapsed += interval

    return "timeout"

@app.post("/upload_and_predict")
async def upload_and_predict(
    clientes: UploadFile = File(...),
    productos: UploadFile = File(...),
    transacciones: UploadFile = File(...)
):
    # 1. Generar fecha de ejecución
    execution_date = datetime.now().strftime("%Y-%m-%d")
    base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", execution_date))
    data_path = os.path.join(base_path, "data")
    os.makedirs(data_path, exist_ok=True)


    # 2. Guardar archivos parquet
    for file, name in zip(
        [clientes, productos, transacciones],
        ["clientes.parquet", "productos.parquet", "transacciones.parquet"]
    ):
        with open(os.path.join(data_path, name), "wb") as f_out:
            shutil.copyfileobj(file.file, f_out)

    # 3. Disparar DAG
    dag_id = "sodAI"
    """
    trigger = subprocess.run([
        "airflow", "dags", "trigger", "-e", execution_date, dag_id
    ], capture_output=True, text=True)

    if trigger.returncode != 0:
        return {"error": f"Error triggering DAG: {trigger.stderr}"}
    """
    # 4. Esperar a que termine con éxito o error
    execution_date_iso = execution_date + "T00:00:00+00:00"
    #state = wait_for_dag_status(dag_id, execution_date_iso)
    state = "success"  # Simulamos éxito para pruebas

    if state == "success":
        execution_date = '2024-12-01'  # Simulamos una fecha de ejecución para pruebas
        base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", execution_date))
        print(f"Base path for predictions: {base_path}")
        prediction_file = os.path.join(base_path, "predictions", "recommended_products.csv")
        if os.path.exists(prediction_file):
            print(f"Prediction file found: {prediction_file}")
            return FileResponse(prediction_file, media_type='text/csv', filename="recommended_products.csv")
        else:
            return {"error": "DAG completed, but predictions file was not found."}

    elif state == "failed":
        return {"error": "Pipeline failed during execution."}

    else:  # timeout
        return {"error": "Pipeline did not complete within the expected time."}
