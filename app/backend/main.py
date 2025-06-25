from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
import os
from datetime import datetime
import shutil
import subprocess

app = FastAPI()

@app.post("/upload_and_predict")
async def upload_and_predict(
    clientes: UploadFile = File(...),
    productos: UploadFile = File(...),
    transacciones: UploadFile = File(...)
):
    # 1. Generar fecha de ejecución
    execution_date = datetime.now().strftime("%Y-%m-%d")
    data_path = os.path.join(execution_date, "data")
    os.makedirs(data_path, exist_ok=True)

    # 2. Guardar archivos parquet
    for file, name in zip(
        [clientes, productos, transacciones],
        ["clientes.parquet", "productos.parquet", "transacciones.parquet"]
    ):
        with open(os.path.join(data_path, name), "wb") as f_out:
            shutil.copyfileobj(file.file, f_out)

    # 3. Disparar pipeline de Airflow
    dag_id = "sodAI"
    trigger = subprocess.run([
        "airflow", "dags", "trigger", "-e", execution_date, dag_id
    ], capture_output=True, text=True)

    if trigger.returncode != 0:
        return {"error": trigger.stderr}

    # 4. Esperar a que termine (puedes reemplazar esto por polling si prefieres)
    # Acá simplemente espera N segundos como placeholder
    import time
    time.sleep(60)  # Esperar ejecución. Reemplazar por verificación real si se desea.

    # 5. Devolver CSV con predicciones
    prediction_file = os.path.join(execution_date, "predicciones.csv")
    if os.path.exists(prediction_file):
        return FileResponse(prediction_file, media_type='text/csv', filename="predicciones.csv")
    else:
        return {"error": "Predicciones no encontradas"}
