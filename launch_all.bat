@echo off
echo 🚀 Starting SodAI Complete System...
echo ======================================

REM Step 0: Setup Docker network and volume
echo 🔧 Step 0: Setting up Docker infrastructure...

REM Check and create network if it doesn't exist
docker network ls | findstr "sodai-network" >nul 2>&1
if %errorlevel%==0 (
    echo    ✅ Network 'sodai-network' already exists
) else (
    echo    Creating Docker network 'sodai-network'...
    docker network create sodai-network
    if %errorlevel%==0 (
        echo    ✅ Network created successfully
    ) else (
        echo    ❌ Failed to create network
    )
)

REM Check and create volume if it doesn't exist
docker volume ls | findstr "sodai-shared-data" >nul 2>&1
if %errorlevel%==0 (
    echo    ✅ Volume 'sodai-shared-data' already exists
) else (
    echo    Creating Docker volume 'sodai-shared-data'...
    docker volume create sodai-shared-data
    if %errorlevel%==0 (
        echo    ✅ Volume created successfully
    ) else (
        echo    ❌ Failed to create volume
    )
)

REM Step 1: Start Airflow
echo 📊 Step 1: Starting Airflow...
cd airflow
docker compose up -d
echo ✅ Airflow starting...

REM Step 2: Start App (Backend + Frontend)
echo 🔧 Step 2: Starting Backend ^& Frontend...
cd ..\app
docker compose up -d
echo ✅ App services starting...

REM Step 3: Wait for services to be ready
echo ⏳ Step 3: Waiting for services to initialize...
timeout /t 10 /nobreak >nul

REM Step 4: Check service health
echo 🔍 Step 4: Checking service health...

REM Check Airflow
echo|set /p="   Airflow (http://localhost:8080): "
curl -s "http://localhost:8080/health" >nul 2>&1
if %errorlevel%==0 (
    echo ✅ Ready
) else (
    echo ⚠️  Starting ^(may take a few more seconds^)
)

REM Check Backend
echo|set /p="   Backend (http://localhost:8000): "
curl -s "http://localhost:8000/" >nul 2>&1
if %errorlevel%==0 (
    echo ✅ Ready
) else (
    echo ❌ Not responding
)

REM Check Frontend
echo|set /p="   Frontend (http://localhost:7860): "
curl -s "http://localhost:7860/" >nul 2>&1
if %errorlevel%==0 (
    echo ✅ Ready
) else (
    echo ❌ Not responding
)

REM Step 5: Check and activate DAG
echo 🔄 Step 5: Checking DAG status...
timeout /t 5 /nobreak >nul

REM Create temp file for curl output
set TEMP_FILE=%TEMP%\dag_status.txt
curl -s "http://localhost:8000/dag/status" > "%TEMP_FILE%" 2>nul

if exist "%TEMP_FILE%" (
    findstr /c:"airflow_available.*true" "%TEMP_FILE%" >nul
    if %errorlevel%==0 (
        findstr /c:"is_paused.*false" "%TEMP_FILE%" >nul
        if %errorlevel%==0 (
            echo    ✅ DAG is active ^(auto-activated by backend^)
        ) else (
            echo    🔄 DAG was paused, activating...
            set TEMP_ACTIVATION=%TEMP%\dag_activation.txt
            curl -s -X POST "http://localhost:8000/dag/activate" > "%TEMP_ACTIVATION%" 2>nul
            if exist "%TEMP_ACTIVATION%" (
                findstr /c:"success.*true" "%TEMP_ACTIVATION%" >nul
                if %errorlevel%==0 (
                    echo    ✅ DAG activated successfully
                ) else (
                    echo    ⚠️  DAG activation failed - you may need to activate manually
                )
                del "%TEMP_ACTIVATION%" >nul 2>&1
            )
        )
    ) else (
        echo    ⚠️  Airflow not ready yet - DAG will auto-activate when ready
    )
    del "%TEMP_FILE%" >nul 2>&1
) else (
    echo    ⚠️  Could not check DAG status - DAG will auto-activate when ready
)

echo.
echo 🎉 SodAI System Status:
echo ======================
echo    Frontend:  http://localhost:7860
echo    Backend:   http://localhost:8000
echo    Airflow:   http://localhost:8080
echo.
echo 📋 Available API endpoints:
echo    GET  /dag/status     - Check DAG status
echo    POST /dag/activate   - Manually activate DAG  
echo    POST /upload_and_start - Start processing
echo.
echo ✨ The system will auto-activate the DAG when ready!
echo    You can now upload parquet files at http://localhost:7860

cd ..
