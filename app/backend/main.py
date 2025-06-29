from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse, JSONResponse
import os
import uuid
import json
from datetime import datetime
import requests
import time
import threading
from typing import Dict, Any
import asyncio

app = FastAPI()
AIRFLOW_BASE_URL = "http://sodai-airflow:8080/api/v2"
DAG_ID = "sodAI"

# In-memory storage for job status and logs
jobs: Dict[str, Dict[str, Any]] = {}

# Detailed task definitions with all DAG tasks
TASK_DEFINITIONS = {
    # Setup tasks (handled by backend)
    "upload": {"name": "📁 Subiendo archivos", "weight": 5, "duration": 30},
    "trigger": {"name": "🚀 Disparando pipeline", "weight": 5, "duration": 20},
    
    # DAG tasks (in execution order)
    "check_data": {"name": "🔍 Verificando datos", "weight": 5, "duration": 15},
    "get_data": {"name": "📊 Obteniendo datos", "weight": 8, "duration": 45},
    "process_data": {"name": "⚙️ Procesando datos", "weight": 10, "duration": 90},
    "holdout": {"name": "🎯 Dividiendo datos (holdout)", "weight": 8, "duration": 60},
    "feature_engineering": {"name": "🔧 Ingeniería de características", "weight": 12, "duration": 120},
    "detect_drift": {"name": "🔍 Detectando drift de datos", "weight": 8, "duration": 45},
    "decide_retraining": {"name": "🤔 Decidiendo estrategia", "weight": 3, "duration": 10},
    "copy_previous_model": {"name": "📋 Copiando modelo previo", "weight": 5, "duration": 30},
    "optimize_hyperparameters": {"name": "🔧 Optimizando hiperparámetros", "weight": 25, "duration": 300},
    "train_model": {"name": "🚀 Entrenando modelo", "weight": 20, "duration": 180},
    "evaluate_model": {"name": "📈 Evaluando modelo", "weight": 8, "duration": 60},
    "export_model": {"name": "💾 Exportando modelo", "weight": 5, "duration": 30},
    "model_ready": {"name": "✅ Modelo listo", "weight": 2, "duration": 5},
    "run_prediction": {"name": "🎯 Generando predicciones", "weight": 12, "duration": 90},
    "get_products": {"name": "🛍️ Obteniendo productos recomendados", "weight": 8, "duration": 45},
    "end": {"name": "🎉 Completado", "weight": 2, "duration": 5}
}

# Task execution order for progress calculation
TASK_ORDER = [
    "upload", "trigger", "check_data", "get_data", "process_data", 
    "holdout", "feature_engineering", "detect_drift", "decide_retraining",
    # Branching: either copy_previous_model OR (optimize_hyperparameters -> train_model -> evaluate_model -> export_model)
    "optimize_hyperparameters", "train_model", "evaluate_model", "export_model",
    "model_ready", "run_prediction", "get_products", "end"
]

def get_total_estimated_time():
    """Calculate total estimated time for all tasks"""
    return sum(task["duration"] for task in TASK_DEFINITIONS.values())

def get_task_display_name(task_id: str) -> str:
    """Get the display name for a task"""
    return TASK_DEFINITIONS.get(task_id, {}).get("name", task_id)

def calculate_progress(completed_tasks, running_tasks, skipped_tasks):
    """Calculate progress based on completed tasks with branching logic"""
    total_weight = 0
    completed_weight = 0
    
    # Determine which branch we're taking
    has_copy_previous = any(t.get('task_id') == 'copy_previous_model' and t.get('state') in ['success', 'running'] for t in completed_tasks + running_tasks)
    has_optimization = any(t.get('task_id') == 'optimize_hyperparameters' and t.get('state') in ['success', 'running', 'failed'] for t in completed_tasks + running_tasks)
    
    # Calculate weights based on actual execution path
    for task_id in TASK_ORDER:
        if task_id in ["upload", "trigger"]:
            # These are always included
            total_weight += TASK_DEFINITIONS[task_id]["weight"]
        elif task_id == "copy_previous_model":
            # Only include if this branch was taken
            if has_copy_previous:
                total_weight += TASK_DEFINITIONS[task_id]["weight"]
        elif task_id in ["optimize_hyperparameters", "train_model", "evaluate_model", "export_model"]:
            # Only include if optimization branch was taken
            if has_optimization or not has_copy_previous:
                total_weight += TASK_DEFINITIONS[task_id]["weight"]
        else:
            # Include all other tasks
            total_weight += TASK_DEFINITIONS[task_id]["weight"]
    
    # Calculate completed weight
    for task in completed_tasks:
        task_id = task.get('task_id')
        if task_id in TASK_DEFINITIONS:
            completed_weight += TASK_DEFINITIONS[task_id]["weight"]
    
    # Add partial weight for running tasks
    for task in running_tasks:
        task_id = task.get('task_id')
        if task_id in TASK_DEFINITIONS:
            # Give 50% credit for running tasks
            completed_weight += TASK_DEFINITIONS[task_id]["weight"] * 0.5
    
    # Calculate progress percentage
    if total_weight > 0:
        progress = min((completed_weight / total_weight) * 100, 100)
    else:
        progress = 0
    
    return max(0, min(100, progress))

def trigger_dag_api(dag_id: str, execution_date: str):
    url = f"{AIRFLOW_BASE_URL}/dags/{dag_id}/dagRuns"
    payload = {"logical_date": execution_date + "T00:00:00+00:00"}
    headers = {"Content-Type": "application/json"}
    resp = requests.post(url, json=payload, headers=headers)
    if resp.status_code in (200, 201):
        return True, resp.json()
    else:
        return False, resp.text

def find_dag_run_by_logical_date(dag_id: str, execution_date: str):
    """Find the actual DAG run by logical date"""
    logical_date = execution_date + "T00:00:00Z"  # ISO format
    
    try:
        # Get all DAG runs for this DAG
        url = f"{AIRFLOW_BASE_URL}/dags/{dag_id}/dagRuns"
        resp = requests.get(url)
        
        if resp.status_code == 200:
            dag_runs = resp.json().get("dag_runs", [])
            
            # Find the run with matching logical_date
            for run in dag_runs:
                if run.get("logical_date") == logical_date:
                    return run.get("dag_run_id")
        
        return None
    except Exception as e:
        print(f"Error finding DAG run: {e}")
        return None

def get_dag_run_status(dag_id: str, execution_date: str):
    """Get DAG run status by logical date"""
    dag_run_id = find_dag_run_by_logical_date(dag_id, execution_date)
    
    if not dag_run_id:
        return None
    
    try:
        url = f"{AIRFLOW_BASE_URL}/dags/{dag_id}/dagRuns/{dag_run_id}"
        resp = requests.get(url)
        
        if resp.status_code == 200:
            return resp.json()
        
        return None
    except Exception as e:
        print(f"Error getting DAG run status: {e}")
        return None

def get_dag_tasks_status(dag_id: str, execution_date: str):
    """Get detailed status of all tasks in the DAG"""
    dag_run_id = find_dag_run_by_logical_date(dag_id, execution_date)
    
    if not dag_run_id:
        return []
    
    try:
        url = f"{AIRFLOW_BASE_URL}/dags/{dag_id}/dagRuns/{dag_run_id}/taskInstances"
        resp = requests.get(url)
        
        if resp.status_code == 200:
            tasks = resp.json().get("task_instances", [])
            return tasks
        
        return []
    except Exception as e:
        print(f"Error getting task status: {e}")
        return []

def get_task_logs(dag_id: str, execution_date: str, task_id: str, try_number: int = 1):
    """Get logs for a specific task"""
    dag_run_id = find_dag_run_by_logical_date(dag_id, execution_date)
    
    if not dag_run_id:
        return ""
    
    try:
        url = f"{AIRFLOW_BASE_URL}/dags/{dag_id}/dagRuns/{dag_run_id}/taskInstances/{task_id}/logs/{try_number}"
        resp = requests.get(url)
        
        if resp.status_code == 200:
            # Parse JSON response to get actual log content
            try:
                log_data = resp.json()
                if 'content' in log_data and log_data['content']:
                    # Extract actual log messages from the structured response
                    log_messages = []
                    for entry in log_data['content']:
                        if 'event' in entry and entry['event'] not in ['::group::', '::endgroup::']:
                            log_messages.append(entry['event'])
                        if 'error_detail' in entry and entry['error_detail']:
                            for error in entry['error_detail']:
                                if 'exc_value' in error:
                                    log_messages.append(f"ERROR: {error['exc_value']}")
                    return '\n'.join(log_messages) if log_messages else resp.text
                else:
                    return resp.text
            except:
                return resp.text
        
        return ""
    except Exception as e:
        return f"Error getting logs: {str(e)}"

def update_job_progress(job_id: str):
    """Background task to update job progress"""
    job = jobs[job_id]
    execution_date = job["execution_date"]
    
    try:
        start_time = time.time()
        total_estimated = get_total_estimated_time()
        
        while job["status"] == "running":
            try:
                # Get DAG run status using the improved function
                dag_data = get_dag_run_status(DAG_ID, execution_date)
                
                if not dag_data:
                    # DAG run not found or not started yet
                    job["current_task"] = "⏳ Esperando inicio de pipeline"
                    time.sleep(10)
                    continue
                
                    dag_state = dag_data.get("state")
                    
                    # Get task details
                    tasks = get_dag_tasks_status(DAG_ID, execution_date)
                    
                    # Update progress based on completed tasks
                    completed_tasks = [t for t in tasks if t.get("state") == "success"]
                    failed_tasks = [t for t in tasks if t.get("state") == "failed"]
                    running_tasks = [t for t in tasks if t.get("state") == "running"]
                    
                    # Check for failed tasks FIRST, even if DAG state is not failed
                    if failed_tasks:
                        job["status"] = "failed"
                        job["current_task"] = "Error en el procesamiento"
                        
                        # Get error details from failed tasks
                        failed_task = failed_tasks[0]
                        task_id = failed_task.get('task_id')
                        job["error"] = f"Error en tarea: {task_id}"
                        job["current_task"] = f"❌ Error en: {task_id}"
                        
                        # Get error logs
                        try:
                            error_logs = get_task_logs(DAG_ID, execution_date, task_id, failed_task.get('try_number', 1))
                            if error_logs and len(error_logs) > 100:
                                # Extract error message from logs
                                if '"ArrowInvalid"' in error_logs and 'Parquet magic bytes' in error_logs:
                                    job["error"] = "❌ Archivo Parquet corrupto o inválido"
                                    job["logs"].append("❌ ERROR: Los archivos parquet están corruptos o no son válidos")
                                    job["logs"].append("💡 TIP: Verifica que los archivos sean parquet válidos")
                                elif '"FileNotFoundError"' in error_logs:
                                    job["error"] = "❌ Archivos no encontrados"
                                    job["logs"].append("❌ ERROR: No se encontraron los archivos de datos")
                                else:
                                    # Show last part of error log
                                    job["logs"].append(f"❌ ERROR: {error_logs[-300:]}")
                        except Exception as e:
                            job["logs"].append(f"❌ Error obteniendo logs: {str(e)}")
                        break
                        
                    # Calculate progress
                    elif dag_state == "success":
                        job["progress"] = 100
                        job["status"] = "completed"
                        job["current_task"] = "¡Completado!"
                        
                        # Try to set the result file path
                        prediction_file = f"/shared-data/{execution_date}/predictions/recommended_products.csv"
                        if os.path.exists(prediction_file):
                            job["result_file"] = prediction_file
                        job["has_result"] = True
                        job["logs"].append("✅ ¡Predicciones generadas exitosamente!")
                        job["logs"].append(f"📊 Archivo de resultados: recommended_products.csv")
                        break
                        
                    elif dag_state in ["failed", "error"]:
                        job["status"] = "failed"
                        job["current_task"] = "Error en el procesamiento"
                        job["error"] = f"DAG falló con estado: {dag_state}"
                        break
                        
                    else:
                        # Still running - use detailed progress tracking
                        skipped_tasks = [t for t in tasks if t.get("state") == "skipped"]
                        
                        # Get current running task with detailed info
                        if running_tasks:
                            current_task = running_tasks[0]
                            current_task_id = current_task.get('task_id', 'unknown')
                            
                            # Get display name for current task
                            display_name = get_task_display_name(current_task_id)
                            job["current_task"] = display_name
                            
                            # Add progress log for task transitions
                        if job.get('_last_task_id') and job.get('_last_task_id') != current_task_id:
                                job["logs"].append(f"▶️ Iniciando: {display_name}")
                        job['_last_task_id'] = current_task_id
                            
                        elif completed_tasks:
                            # Show last completed task if nothing is running
                            last_completed = completed_tasks[-1]
                            last_task_id = last_completed.get('task_id', 'unknown')
                            display_name = get_task_display_name(last_task_id)
                            job["current_task"] = f"✅ Completado: {display_name}"
                        
                        # Calculate progress using new weighted system
                        progress = calculate_progress(completed_tasks, running_tasks, skipped_tasks)
                        job["progress"] = max(job.get("progress", 0), progress)  # Never decrease progress
                        
                        # Enhanced time estimation
                        elapsed = time.time() - start_time
                        if job["progress"] > 5:  # Only estimate after some progress
                            estimated_total = (elapsed / job["progress"]) * 100
                            remaining = max(0, estimated_total - elapsed)
                            job["estimated_remaining"] = int(remaining)
                        
                        # Add detailed task completion logs
                        for task in completed_tasks:
                            task_id = task.get('task_id')
                            if task_id and not job.get(f"_logged_{task_id}", False):
                                display_name = get_task_display_name(task_id)
                                job["logs"].append(f"✅ {display_name}")
                                job[f"_logged_{task_id}"] = True
                        
            except Exception as e:
                job["logs"].append(f"Error monitoring progress: {str(e)}")
                print(f"Error in update_job_progress: {e}")
            
            time.sleep(10)  # Check every 10 seconds
            
    except Exception as e:
        job["status"] = "failed"
        job["error"] = f"Monitoring error: {str(e)}"

@app.post("/upload_and_start")
async def upload_and_start(
    clientes: UploadFile = File(...),
    productos: UploadFile = File(...),
    transacciones: UploadFile = File(...)
):
    """Upload files and start processing asynchronously"""
    
    # Generate unique job ID
    job_id = str(uuid.uuid4())
    execution_date = datetime.now().strftime("%Y-%m-%d")
    
    # Initialize job status
    jobs[job_id] = {
        "status": "running",
        "progress": 0,
        "current_task": TASK_DEFINITIONS["upload"]["name"],
        "logs": [],
        "execution_date": execution_date,
        "estimated_remaining": get_total_estimated_time(),
        "created_at": datetime.now().isoformat()
    }
    
    try:
        # Save uploaded files
        base_path = f"/shared-data/{execution_date}"
        data_path = os.path.join(base_path, "data")
        os.makedirs(data_path, exist_ok=True)
        
        jobs[job_id]["logs"].append("📁 Guardando archivos parquet...")
        
        for file, name in zip(
            [clientes, productos, transacciones],
            ["clientes.parquet", "productos.parquet", "transacciones.parquet"]
        ):
            await file.seek(0)
            content = await file.read()
            
            # Verify it's actually a parquet file
            if len(content) < 4:
                jobs[job_id]["status"] = "failed"
                jobs[job_id]["error"] = f"El archivo {name} es demasiado pequeño ({len(content)} bytes)"
                return {"job_id": job_id, "status": "failed", "error": jobs[job_id]["error"]}
            
            # Check magic bytes
            if not content.startswith(b'PAR1'):
                first_bytes = content[:8].hex() if len(content) >= 8 else content.hex()
                jobs[job_id]["status"] = "failed"
                jobs[job_id]["error"] = f"El archivo {name} no es un archivo Parquet válido. Magic bytes: {first_bytes} (esperado: 50415231)"
                jobs[job_id]["logs"].append(f"❌ {name}: Magic bytes incorrectos. Archivo: {first_bytes}, Esperado: 50415231 (PAR1)")
                return {"job_id": job_id, "status": "failed", "error": jobs[job_id]["error"]}
            
            file_path = os.path.join(data_path, name)
            with open(file_path, "wb") as f_out:
                f_out.write(content)
            
            # Verify file was written correctly
            if not os.path.exists(file_path) or os.path.getsize(file_path) != len(content):
                jobs[job_id]["status"] = "failed"
                jobs[job_id]["error"] = f"Error al guardar {name}"
                return {"job_id": job_id, "status": "failed", "error": jobs[job_id]["error"]}
                
            jobs[job_id]["logs"].append(f"✅ {name} guardado exitosamente ({len(content):,} bytes)")
        
        # Update progress after upload
        upload_weight = TASK_DEFINITIONS["upload"]["weight"]
        total_weight = sum(task["weight"] for task in TASK_DEFINITIONS.values())
        jobs[job_id]["progress"] = int((upload_weight / total_weight) * 100)
        jobs[job_id]["current_task"] = TASK_DEFINITIONS["trigger"]["name"]
        jobs[job_id]["logs"].append("🚀 Iniciando pipeline de procesamiento...")
        
        # Ensure DAG is active before triggering
        if not ensure_dag_active():
            jobs[job_id]["status"] = "failed"
            jobs[job_id]["error"] = "No se pudo activar el DAG de Airflow"
            return {"job_id": job_id, "status": "failed", "error": jobs[job_id]["error"]}
        
        jobs[job_id]["logs"].append("✅ DAG verificado y activo")
        
        # Trigger DAG
        success, resp = trigger_dag_api(DAG_ID, execution_date)
        if not success:
            jobs[job_id]["status"] = "failed"
            jobs[job_id]["error"] = f"Failed to trigger DAG: {resp}"
            return {"job_id": job_id, "status": "failed", "error": jobs[job_id]["error"]}
        
        # Update progress after triggering
        trigger_weight = upload_weight + TASK_DEFINITIONS["trigger"]["weight"]
        jobs[job_id]["progress"] = int((trigger_weight / total_weight) * 100)
        jobs[job_id]["current_task"] = "⏳ Esperando inicio de pipeline"
        jobs[job_id]["logs"].append("✅ Pipeline disparado exitosamente")
        jobs[job_id]["_logged_upload"] = True
        jobs[job_id]["_logged_trigger"] = True
        
        # Start background monitoring
        thread = threading.Thread(target=update_job_progress, args=(job_id,))
        thread.daemon = True
        thread.start()
        
        return {"job_id": job_id, "status": "started"}
        
    except Exception as e:
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = str(e)
        return {"job_id": job_id, "status": "failed", "error": str(e)}

@app.get("/job/{job_id}/status")
async def get_job_status(job_id: str):
    """Get current job status and progress"""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs[job_id]
    return {
        "job_id": job_id,
        "status": job["status"],
        "progress": job["progress"],
        "current_task": job["current_task"],
        "estimated_remaining": job.get("estimated_remaining", 0),
        "has_result": "result_file" in job,
        "error": job.get("error")
    }

@app.get("/job/{job_id}/logs")
async def get_job_logs(job_id: str):
    """Get current job logs"""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return {"logs": jobs[job_id]["logs"]}

@app.get("/job/{job_id}/download")
async def download_result(job_id: str):
    """Download the result file when job is completed"""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs[job_id]
    if job["status"] != "completed":
        raise HTTPException(status_code=400, detail="Job not completed yet")
    
    if "result_file" not in job:
        raise HTTPException(status_code=404, detail="Result file not found")
    
    result_file = job["result_file"]
    if not os.path.exists(result_file):
        raise HTTPException(status_code=404, detail="Result file not found on disk")
    
    return FileResponse(
        result_file,
        media_type="text/csv",
        filename="recommended_products.csv"
    )

@app.get("/")
async def root():
    return {"message": "SodAI Backend API", "active_jobs": len(jobs)}

@app.get("/dag/status")
async def get_dag_status():
    """Get current DAG status and manage it"""
    try:
        # Check DAG status
        dag_response = requests.get(f"{AIRFLOW_BASE_URL}/dags/{DAG_ID}")
        if dag_response.status_code == 200:
            dag_data = dag_response.json()
            return {
                "dag_id": DAG_ID,
                "is_paused": dag_data.get("is_paused", True),
                "is_stale": dag_data.get("is_stale", False),
                "last_parsed_time": dag_data.get("last_parsed_time"),
                "airflow_available": True
            }
        else:
            return {
                "dag_id": DAG_ID,
                "airflow_available": False,
                "error": f"Airflow response: {dag_response.status_code}"
            }
    except Exception as e:
        return {
            "dag_id": DAG_ID,
            "airflow_available": False,
            "error": str(e)
        }

@app.post("/dag/activate")
async def activate_dag():
    """Manually activate the DAG"""
    if ensure_dag_active():
        return {"message": f"DAG '{DAG_ID}' activated successfully", "success": True}
    else:
        return {"message": f"Failed to activate DAG '{DAG_ID}'", "success": False}

@app.get("/system/status")
async def get_system_status():
    """Get complete system status for debugging"""
    try:
        # Check Airflow
        airflow_status = "down"
        dag_status = "unknown"
        try:
            airflow_response = requests.get(f"{AIRFLOW_BASE_URL}/health", timeout=5)
            if airflow_response.status_code == 200:
                airflow_status = "up"
                
                # Check DAG status
                dag_response = requests.get(f"{AIRFLOW_BASE_URL}/dags/{DAG_ID}", timeout=5)
                if dag_response.status_code == 200:
                    dag_data = dag_response.json()
                    dag_status = "paused" if dag_data.get("is_paused", True) else "active"
                else:
                    dag_status = "not_found"
        except:
            airflow_status = "down"
        
        # Check recent DAG runs
        recent_runs = []
        try:
            runs_response = requests.get(f"{AIRFLOW_BASE_URL}/dags/{DAG_ID}/dagRuns?limit=3", timeout=5)
            if runs_response.status_code == 200:
                runs_data = runs_response.json()
                for run in runs_data.get("dag_runs", []):
                    recent_runs.append({
                        "run_id": run.get("dag_run_id"),
                        "state": run.get("state"),
                        "logical_date": run.get("logical_date"),
                        "start_date": run.get("start_date")
                    })
        except:
            pass
        
        return {
            "timestamp": datetime.now().isoformat(),
            "backend": {
                "status": "up",
                "active_jobs": len(jobs),
                "job_ids": list(jobs.keys())
            },
            "airflow": {
                "status": airflow_status,
                "url": f"{AIRFLOW_BASE_URL.replace('api/v2', '')}"
            },
            "dag": {
                "id": DAG_ID,
                "status": dag_status,
                "recent_runs": recent_runs
            },
            "endpoints": {
                "frontend": "http://localhost:7860",
                "backend": "http://localhost:8000",
                "airflow": "http://localhost:8080"
            }
        }
        
    except Exception as e:
        return {
            "timestamp": datetime.now().isoformat(),
            "error": str(e),
            "backend": {"status": "up", "active_jobs": len(jobs)}
        }

# Cleanup old jobs periodically (optional)
@app.on_event("startup")
async def startup_event():
    print("🚀 SodAI Backend started with progress tracking!")
    
    # Auto-unpause the DAG if it's paused
    try:
        print("🔍 Checking DAG status...")
        
        # Check if DAG is paused
        dag_response = requests.get(f"{AIRFLOW_BASE_URL}/dags/{DAG_ID}")
        if dag_response.status_code == 200:
            dag_data = dag_response.json()
            is_paused = dag_data.get("is_paused", True)
            
            if is_paused:
                print(f"⚠️  DAG '{DAG_ID}' is paused. Activating automatically...")
                
                # Unpause the DAG
                unpause_response = requests.patch(
                    f"{AIRFLOW_BASE_URL}/dags/{DAG_ID}",
                    json={"is_paused": False},
                    headers={"Content-Type": "application/json"}
                )
                
                if unpause_response.status_code == 200:
                    print(f"✅ DAG '{DAG_ID}' activated successfully!")
                else:
                    print(f"❌ Failed to activate DAG: {unpause_response.status_code} - {unpause_response.text}")
            else:
                print(f"✅ DAG '{DAG_ID}' is already active")
        else:
            print(f"⚠️  Could not check DAG status: {dag_response.status_code} - {dag_response.text}")
            print("💡 DAG might not be available yet, will retry on first job")
            
    except Exception as e:
        print(f"⚠️  Error during DAG check: {str(e)}")
        print("💡 Will retry activating DAG on first job submission")

def ensure_dag_active():
    """Ensure DAG is active before triggering"""
    try:
        # Check if DAG is paused
        dag_response = requests.get(f"{AIRFLOW_BASE_URL}/dags/{DAG_ID}")
        if dag_response.status_code == 200:
            dag_data = dag_response.json()
            is_paused = dag_data.get("is_paused", True)
            
            if is_paused:
                print(f"🔄 DAG '{DAG_ID}' is paused. Activating...")
                
                # Unpause the DAG
                unpause_response = requests.patch(
                    f"{AIRFLOW_BASE_URL}/dags/{DAG_ID}",
                    json={"is_paused": False},
                    headers={"Content-Type": "application/json"}
                )
                
                if unpause_response.status_code == 200:
                    print(f"✅ DAG '{DAG_ID}' activated successfully!")
                    return True
                else:
                    print(f"❌ Failed to activate DAG: {unpause_response.status_code}")
                    return False
            else:
                return True
        else:
            print(f"❌ Could not check DAG status: {dag_response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Error checking DAG status: {str(e)}")
        return False
 