@echo off
setlocal

echo Creating shared network and volume...
docker network create sodai-network 2>nul
docker volume create sodai-shared-data

echo Building Airflow...
cd airflow
docker compose build

echo Building App...
cd ..\app
docker compose build

echo Starting containers...
docker compose up -d

cd ..\airflow
docker compose up -d

echo.
echo Frontend: http://localhost:7860
echo Backend:  http://localhost:8000
echo Airflow:  http://localhost:8080
pause
