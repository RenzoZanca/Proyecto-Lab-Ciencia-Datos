FROM python:3.10-slim

# Variables de entorno de Airflow 3.0.1
ENV AIRFLOW_HOME=/app/airflow
ENV AIRFLOW__CORE__DAGS_FOLDER=/app/airflow/dags
ENV AIRFLOW__CORE__LOAD_EXAMPLES=False
ENV AIRFLOW__CORE__EXECUTOR=LocalExecutor
ENV AIRFLOW__CORE__SIMPLE_AUTH_MANAGER_ALL_ADMINS=True

WORKDIR ${AIRFLOW_HOME}

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copiar requirements e instalar
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Inicializar Airflow
RUN mkdir -p "${AIRFLOW_HOME}/dags" \
    && mkdir -p "${AIRFLOW_HOME}/plugins" \
    && mkdir -p "${AIRFLOW_HOME}/logs" \
    && airflow db migrate

# Copiar nuestros archivos del proyecto
COPY dag.py "${AIRFLOW_HOME}/dags/"
COPY data_functions.py "${AIRFLOW_HOME}/dags/"
COPY train_functions.py "${AIRFLOW_HOME}/dags/"
COPY predictions_functions.py "${AIRFLOW_HOME}/dags/"
COPY plugins/ "${AIRFLOW_HOME}/plugins/"


EXPOSE 8080
CMD ["airflow", "standalone"] 
