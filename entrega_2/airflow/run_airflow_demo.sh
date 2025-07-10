#!/bin/bash

echo "🚀 SodAI Drinks - Airflow Demo Setup"
echo "===================================="
echo

# Limpiar contenedores anteriores
echo "🧹 Limpiando contenedores anteriores..."
docker-compose down 2>/dev/null
docker container rm sodai-airflow 2>/dev/null

# Construir imagen
echo "🔨 Construyendo imagen de Docker..."
docker-compose build

if [ $? -ne 0 ]; then
    echo "❌ Error en la construcción de Docker"
    exit 1
fi

# Iniciar contenedor
echo "🎬 Iniciando Airflow para demostración..."
docker-compose up -d

echo
echo "⏳ Esperando que Airflow esté listo..."
echo "   Esto puede tomar 1-2 minutos..."

# Esperar que el contenedor esté healthy
for i in {1..40}; do
    if docker exec sodai-airflow curl -s http://localhost:8080/health > /dev/null 2>&1; then
        echo "✅ Airflow está listo!"
        break
    fi
    if [ $i -eq 40 ]; then
        echo "⚠️  Airflow está tardando más de lo esperado..."
        echo "   Verifica los logs con: docker-compose logs"
    fi
    echo "   Esperando... ($i/40)"
    sleep 3
done

echo
echo "🎊 ¡Airflow está ejecutándose!"
echo "📺 Interfaz web: http://localhost:8080"
echo "👤 Usuario: admin"
echo "🔑 Contraseña: admin"
echo
echo "🎬 Para el video:"
echo "   1. Ve a http://localhost:8080"
echo "   2. Inicia sesión con admin/admin"
echo "   3. Busca el DAG 'sodAI'"
echo "   4. Actívalo y ejecuta manualmente"
echo "   5. ¡Graba la ejecución!"
echo
echo "🛑 Para detener: docker-compose down" 