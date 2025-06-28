Crear una red (ejecutar una sola vez)

- docker network create sodai-network
- docker volume create sodai-shared-data
 
Levantar contenedores en 2 terminales separadas:
- cd airflow
- docker compose build
- docker compose up

- cd app
- docker compose build
- docker compose up