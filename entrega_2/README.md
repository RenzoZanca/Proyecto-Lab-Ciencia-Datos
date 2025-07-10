# 🥤 SodAI - Sistema de Recomendación de Productos

Sistema MLOps completo para generar recomendaciones de productos usando Airflow, FastAPI y Gradio.

## 🚀 Inicio Rápido

### Prerrequisitos
- Docker Desktop instalado y ejecutándose
- Al menos 4GB de RAM disponible para Docker

### Lanzar el Sistema Completo

```bash
./launch_all.sh
```

Este script automáticamente:
- ✅ Crea la red Docker necesaria
- ✅ Inicia Airflow (pipeline ML)
- ✅ Inicia el Backend (API)
- ✅ Inicia el Frontend (interfaz web)
- ✅ Activa el DAG automáticamente

### Acceder al Sistema

Una vez iniciado, puedes acceder a:

| Servicio | URL | Descripción |
|----------|-----|-------------|
| **Frontend** | http://localhost:7860 | Interfaz principal para subir archivos |
| **Backend API** | http://localhost:8000 | API REST para gestión de trabajos |
| **Airflow** | http://localhost:8080 | Dashboard del pipeline ML |

## 📁 Usar el Sistema

1. **Accede al frontend**: http://localhost:7860
2. **Sube los archivos parquet**:
   - `clientes.parquet`
   - `productos.parquet` 
   - `transacciones.parquet`
3. **Haz clic en "Subir Archivos e Iniciar Procesamiento"**
4. **Observa el progreso** en tiempo real
5. **Descarga o visualiza los resultados** cuando termine

## 🔧 Herramientas de Diagnóstico

### Si tienes problemas de conectividad:

```bash
./fix_connectivity.sh
```

### Para diagnóstico detallado:

```bash
./diagnose_system.sh
```

### Ver guía completa de solución de problemas:

```bash
cat SOLUCION_ERRORES_CONECTIVIDAD.md
```

## 🏗️ Arquitectura del Sistema

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Frontend      │    │    Backend      │    │    Airflow      │
│   (Gradio)      │────│   (FastAPI)     │────│   (Pipeline)    │
│   Port: 7860    │    │   Port: 8000    │    │   Port: 8080    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                              │
                              ▼
                       ┌─────────────────┐
                       │ Shared Volume   │
                       │ (Data & Models) │
                       └─────────────────┘
```

### Componentes

1. **Frontend (Gradio)**: Interfaz web para cargar archivos y visualizar resultados
2. **Backend (FastAPI)**: API REST que gestiona trabajos y se comunica con Airflow
3. **Airflow**: Ejecuta el pipeline ML con todas las tareas de procesamiento
4. **Volumen compartido**: Almacena datos y modelos entre servicios

## 📊 Pipeline ML

El sistema ejecuta automáticamente:

1. **🔍 Verificación de datos** - Valida formato parquet
2. **📊 Obtención de datos** - Carga los archivos
3. **⚙️ Procesamiento** - Limpieza y preparación
4. **🎯 División holdout** - Separa datos de entrenamiento/prueba
5. **🔧 Ingeniería de características** - Crea features para ML
6. **📈 Detección de drift** - Analiza cambios en los datos
7. **🤔 Decisión de reentrenamiento** - Decide si entrenar modelo nuevo
8. **🚀 Optimización de hiperparámetros** - Busca mejores parámetros
9. **🎯 Entrenamiento del modelo** - Entrena modelo final
10. **📈 Evaluación** - Valida performance
11. **💾 Exportación** - Guarda modelo entrenado
12. **🎯 Generación de predicciones** - Crea recomendaciones
13. **🛍️ Obtención de productos** - Prepara resultados finales

## 📈 Características

- ✅ **Interfaz amigable** con Gradio
- ✅ **Progreso en tiempo real** con estimaciones de tiempo
- ✅ **Validación automática** de archivos parquet
- ✅ **Optimización de memoria** para evitar crashes
- ✅ **Logging detallado** de cada paso
- ✅ **Visualización de resultados** directa en la interfaz
- ✅ **Descarga de resultados** en CSV
- ✅ **Estadísticas automáticas** de las predicciones

## 🛠️ Comandos Útiles

### Gestión del Sistema

```bash
# Inicio completo
./launch_all.sh

# Diagnóstico
./diagnose_system.sh

# Reparación rápida
./fix_connectivity.sh

# Parar todos los servicios
docker-compose -f app/docker-compose.yml down
docker-compose -f airflow/docker-compose.yml down
```

### Monitoreo

```bash
# Ver contenedores ejecutándose
docker ps

# Ver logs de servicios
docker-compose -f airflow/docker-compose.yml logs -f
docker-compose -f app/docker-compose.yml logs -f

# Estado del sistema
curl http://localhost:8000/system/status
```

### Limpieza

```bash
# Limpieza completa
docker-compose -f app/docker-compose.yml down
docker-compose -f airflow/docker-compose.yml down
docker network rm sodai-network
docker volume rm sodai-shared-data
```

## 🚨 Solución de Problemas

### Errores Comunes

| Error | Causa | Solución |
|-------|-------|----------|
| `NameResolutionError: Failed to resolve 'sodai-airflow'` | Contenedores no en misma red | `./fix_connectivity.sh` |
| `Could not check DAG status: 401` | Airflow inicializando | Esperar 2-3 minutos |
| `No se pudo activar el DAG` | Backend no conecta a Airflow | `./fix_connectivity.sh` |
| `Error de memoria` en optimización | Pocos recursos | Aumentar RAM de Docker |

### Verificaciones Rápidas

```bash
# ¿Están todos los servicios ejecutándose?
docker ps | grep sodai

# ¿Pueden los servicios comunicarse?
docker exec sodai-backend curl http://sodai-airflow:8080/health

# ¿Están en la misma red?
docker network inspect sodai-network
```

## 📚 Documentación Adicional

- [Guía de Solución de Errores](SOLUCION_ERRORES_CONECTIVIDAD.md)
- [Documentación Técnica](documentation.md)
- [Conclusiones del Proyecto](conclusiones.md)

## 🔧 Desarrollo

### Estructura del Proyecto

```
Proyecto-Lab-Ciencia-Datos/
├── airflow/                # Pipeline ML
│   ├── dag.py             # Definición del DAG
│   ├── data_functions.py  # Funciones de datos
│   ├── train_functions.py # Funciones de entrenamiento
│   └── docker-compose.yml
├── app/                   # Aplicación web
│   ├── backend/           # API FastAPI
│   │   └── main.py
│   ├── frontend/          # Interfaz Gradio
│   │   └── app.py
│   └── docker-compose.yml
├── launch_all.sh          # Script de inicio
├── diagnose_system.sh     # Herramienta de diagnóstico
└── fix_connectivity.sh    # Solución rápida
```

### APIs Disponibles

| Endpoint | Método | Descripción |
|----------|--------|-------------|
| `/upload_and_start` | POST | Sube archivos e inicia procesamiento |
| `/job/{job_id}/status` | GET | Estado de un trabajo |
| `/job/{job_id}/logs` | GET | Logs de un trabajo |
| `/job/{job_id}/download` | GET | Descarga resultados |
| `/dag/status` | GET | Estado del DAG |
| `/dag/activate` | POST | Activa el DAG |
| `/system/status` | GET | Estado del sistema |

## 🎯 Resultados

El sistema genera:

- **Archivo CSV** con recomendaciones de productos
- **Estadísticas detalladas** en la interfaz web:
  - Total de recomendaciones
  - Clientes únicos con recomendaciones
  - Productos únicos recomendados
  - Estadísticas de probabilidad
  - Top 5 clientes y productos

## 📊 Performance

- **Optimización de memoria** para datasets grandes
- **Paralelización** de tareas ML
- **Límites de memoria** configurables
- **Logging eficiente** sin sobrecargar el sistema

## 🤝 Contribuir

1. Haz fork del proyecto
2. Crea una rama para tu feature
3. Commit tus cambios
4. Push a la rama
5. Abre un Pull Request

## 📄 Licencia

Este proyecto está bajo la licencia MIT.