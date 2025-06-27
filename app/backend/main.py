from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
import os
from datetime import datetime
import shutil
import requests
import time

app = FastAPI()

AIRFLOW_BASE_URL = "http://localhost:8080/api/v1"
DAG_ID = "sodAI"

def trigger_dag_api(dag_id: str, execution_date: str):
    url = f"{AIRFLOW_BASE_URL}/dags/{dag_id}/dagRuns"
    payload = {"execution_date": execution_date + "T00:00:00+00:00"}
    headers = {"Content-Type": "application/json"}
    resp = requests.post(url, json=payload, headers=headers)
    if resp.status_code in (200, 201):
        return True, resp.json()
    else:
        return False, resp.text

def wait_for_dag_status(dag_id: str, execution_date: str, timeout=2400, interval=60):
    url = f"{AIRFLOW_BASE_URL}/dags/{dag_id}/dagRuns/{execution_date}T00:00:00+00:00"
    elapsed = 0
    while elapsed < timeout:
        resp = requests.get(url)
        if resp.status_code == 200:
            state = resp.json().get("state")
            print(f"[{elapsed}s] DAG state: {state}")
            if state == "success":
                return "success"
            elif state in ["failed", "error"]:
                return "failed"
        else:
            print(f"Error getting DAG state: {resp.status_code}")
        time.sleep(interval)
        elapsed += interval
    return "timeout"

@app.post("/upload_and_predict")
async def upload_and_predict(
    clientes: UploadFile = File(...),
    productos: UploadFile = File(...),
    transacciones: UploadFile = File(...)
):
    # Generar fecha de ejecución
    execution_date = datetime.now().strftime("%Y-%m-%d")
    base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", execution_date))
    data_path = os.path.join(base_path, "data")
    os.makedirs(data_path, exist_ok=True)

    # Guardar archivos parquet
    for file, name in zip(
        [clientes, productos, transacciones],
        ["clientes.parquet", "productos.parquet", "transacciones.parquet"]
    ):
        with open(os.path.join(data_path, name), "wb") as f_out:
            shutil.copyfileobj(file.file, f_out)

    # Disparar DAG usando API REST
    success, resp = trigger_dag_api(DAG_ID, execution_date)
    if not success:
        return {"error": f"Failed to trigger DAG: {resp}"}

    # Esperar a que termine el DAG
    state = wait_for_dag_status(DAG_ID, execution_date)

    if state == "success":
        prediction_file = os.path.join(base_path, "predictions", "recommended_products.csv")
        if os.path.exists(prediction_file):
            return FileResponse(prediction_file, media_type='text/csv', filename="recommended_products.csv")
        else:
            return {"error": "DAG completed, but predictions file was not found."}
    elif state == "failed":
        return {"error": "Pipeline failed during execution."}
    else:
        return {"error": "Pipeline did not complete within the expected time."}
