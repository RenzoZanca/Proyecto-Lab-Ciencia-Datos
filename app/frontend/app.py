import gradio as gr
import requests
import pandas as pd
from io import StringIO
import time

# Backend URL
BACKEND_URL = "http://backend:8000"

def format_time(seconds):
    """Format seconds into human readable time"""
    if seconds <= 0:
        return "0s"
    
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"

def subir_archivos_y_iniciar(clientes, productos, transacciones):
    """Upload files and start processing"""
    
    if not clientes or not productos or not transacciones:
        return (
            "❌ Error: Por favor sube los tres archivos requeridos",
            gr.update(visible=False),  # progress_box
            gr.update(visible=False),  # logs_box
            gr.update(visible=False),  # download_box
            "",  # progress_text
            0,   # progress_bar
            "",  # logs_text
            "",  # download_link
            "",  # job_id
            None,  # results_display
            ""   # stats_display
        )
    
    try:
        # Upload files - properly open and read file content
        files = {}
        
        for file_obj, name in zip(
            [clientes, productos, transacciones],
            ["clientes", "productos", "transacciones"]
        ):
            # Check if file_obj is a string (path) or file object
            if isinstance(file_obj, str):
                # It's a file path, open and read it
                with open(file_obj, 'rb') as f:
                    files[name] = (f"{name}.parquet", f.read(), "application/octet-stream")
            else:
                # It's already a file object, read it
                if hasattr(file_obj, 'name'):
                    # It's a file-like object with a name
                    with open(file_obj.name, 'rb') as f:
                        files[name] = (f"{name}.parquet", f.read(), "application/octet-stream")
                else:
                    # It's raw content
                    files[name] = (f"{name}.parquet", file_obj, "application/octet-stream")
        
        print(f"📤 Enviando archivos: {list(files.keys())}")
        for name, (filename, content, mime) in files.items():
            print(f"   {name}: {len(content):,} bytes")
        
        response = requests.post(f"{BACKEND_URL}/upload_and_start", files=files)
        
        if response.status_code != 200:
            return (
                f"❌ Error al subir archivos: {response.text}",
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=False),
                "", 0, "", "", "", None, ""
            )
        
        result = response.json()
        
        if result.get("status") == "failed":
            return (
                f"❌ Error: {result.get('error', 'Error desconocido')}",
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=False),
                "", 0, "", "", "", None, ""
            )
        
        job_id = result["job_id"]
        
        return (
            "✅ Archivos subidos exitosamente. Procesamiento iniciado... (Haz clic en '🔄 Actualizar' para ver el progreso)",
            gr.update(visible=True),   # progress_box
            gr.update(visible=True),   # logs_box
            gr.update(visible=False),  # download_box (hide until complete)
            "🚀 Iniciando procesamiento...",  # progress_text
            5,        # progress_bar
            "🚀 Procesamiento iniciado...\n\n💡 Tip: Haz clic en 'Actualizar Progreso' cada pocos segundos para ver el estado actual.",  # logs_text
            "",                        # download_link
            job_id,                    # job_id
            None,                      # results_display
            ""                         # stats_display
        )
        
    except Exception as e:
        return (
            f"❌ Error de conexión: {str(e)}",
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            "", 0, "", "", "", None, ""
        )

def actualizar_progreso(job_id):
    """Update progress and logs for a job"""
    
    if not job_id:
        return (
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            "No hay trabajo activo",
            0,
            "No hay logs disponibles",
            "",
            None,  # results_display
            ""     # stats_text
        )
    
    try:
        # Get job status
        status_response = requests.get(f"{BACKEND_URL}/job/{job_id}/status")
        if status_response.status_code != 200:
            return (
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=False),
                "Error obteniendo estado",
                0,
                "Error de conexión",
                "",
                None,
                ""
            )
        
        status = status_response.json()
        
        # Get logs
        logs_response = requests.get(f"{BACKEND_URL}/job/{job_id}/logs")
        logs_text = "No hay logs disponibles"
        if logs_response.status_code == 200:
            logs_data = logs_response.json()
            if logs_data.get("logs"):
                logs_text = "\n".join(logs_data["logs"])
                logs_text += f"\n\n⏰ Última actualización: {time.strftime('%H:%M:%S')}"
        
        # Format progress text
        progress_val = status.get("progress", 0)
        current_task = status.get("current_task", "Procesando...")
        estimated_remaining = status.get("estimated_remaining", 0)
        
        if status.get("status") == "completed":
            progress_text = f"✅ {current_task} - 100% completado"
            
            # Automatically load and show results
            download_url = f"{BACKEND_URL}/job/{job_id}/download"
            results_df, stats_text = cargar_y_mostrar_resultados(download_url)
            
            # Show download box
            return (
                gr.update(visible=True),   # progress_box
                gr.update(visible=True),   # logs_box
                gr.update(visible=True),   # download_box
                progress_text,
                100,
                logs_text,
                download_url,
                results_df,  # results_display
                stats_text   # stats_text
            )
            
        elif status.get("status") == "failed":
            error_msg = status.get("error", "Error desconocido")
            progress_text = f"❌ Error: {error_msg}"
            
            return (
                gr.update(visible=True),
                gr.update(visible=True),
                gr.update(visible=False),
                progress_text,
                progress_val,
                logs_text,
                "",
                None,
                ""
            )
        else:
            # Still running
            time_str = format_time(estimated_remaining)
            progress_text = f"🔄 {current_task} - {progress_val:.1f}% - Tiempo estimado: {time_str}"
            
            return (
                gr.update(visible=True),
                gr.update(visible=True),
                gr.update(visible=False),
                progress_text,
                progress_val,
                logs_text,
                "",
                None,
                ""
            )
            
    except Exception as e:
        return (
            gr.update(visible=True),
            gr.update(visible=True),
            gr.update(visible=False),
            f"Error de conexión: {str(e)}",
            0,
            f"Error: {str(e)}",
            "",
            None,
            ""
        )

def cargar_y_mostrar_resultados(download_url):
    """Load results and generate statistics"""
    if not download_url:
        return None, "No hay resultados disponibles"
    
    try:
        response = requests.get(download_url)
        if response.status_code == 200:
            # Parse CSV
                df = pd.read_csv(StringIO(response.text))
            
            if len(df) == 0:
                return None, "❌ No se generaron predicciones"
            
            # Generate statistics
            total_recomendaciones = len(df)
            clientes_unicos = df['customer_id'].nunique()
            productos_unicos = df['product_id'].nunique()
            semana_objetivo = df['week'].iloc[0] if 'week' in df.columns else 'N/A'
            
            prob_promedio = df['probability'].mean()
            prob_min = df['probability'].min()
            prob_max = df['probability'].max()
            
            # Top customers by number of recommendations
            top_customers = df['customer_id'].value_counts().head(5)
            
            # Top products by number of recommendations
            top_products = df['product_id'].value_counts().head(5)
            
            # Generate statistics text
            stats_text = f"""
## 📊 Estadísticas de Predicciones

### 🎯 Resumen General
- **Total de recomendaciones:** {total_recomendaciones:,}
- **Clientes únicos:** {clientes_unicos:,}
- **Productos únicos:** {productos_unicos:,}
- **Semana objetivo:** {semana_objetivo}

### 📈 Probabilidades
- **Promedio:** {prob_promedio:.4f} ({prob_promedio*100:.2f}%)
- **Mínima:** {prob_min:.4f} ({prob_min*100:.2f}%)
- **Máxima:** {prob_max:.4f} ({prob_max*100:.2f}%)

### 🏆 Top 5 Clientes (más recomendaciones)
{chr(10).join([f"- Cliente {customer_id}: {count} productos" for customer_id, count in top_customers.items()])}

### 🛍️ Top 5 Productos (más recomendados)
{chr(10).join([f"- Producto {product_id}: {count} clientes" for product_id, count in top_products.items()])}
            """
            
            # Prepare display DataFrame (show top 100 with highest probability)
            display_df = df.nlargest(100, 'probability').round({'probability': 4})
            
            return display_df, stats_text
            
        else:
            return None, f"❌ Error al cargar resultados: {response.status_code}"
            
    except Exception as e:
        return None, f"❌ Error procesando resultados: {str(e)}"

def descargar_resultados(download_url):
    """Download results manually when button is clicked"""
    results_df, stats_text = cargar_y_mostrar_resultados(download_url)
    return results_df

# Create the Gradio interface
with gr.Blocks(title="SodAI Drinks 🥤", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# SodAI Drinks 🥤")
    gr.Markdown("Sube los archivos parquet para obtener predicciones de compra con seguimiento en tiempo real")
    
    # Job ID storage (hidden)
    job_id_state = gr.State("")
    
    # File upload section
    with gr.Row():
        clientes_file = gr.File(label="📊 Clientes.parquet", file_types=[".parquet"])
        productos_file = gr.File(label="🛍️ Productos.parquet", file_types=[".parquet"])
        transacciones_file = gr.File(label="💳 Transacciones.parquet", file_types=[".parquet"])
    
    # Upload button
    upload_btn = gr.Button("🚀 Subir Archivos e Iniciar Procesamiento", variant="primary", size="lg")
    
    # Status message
    status_msg = gr.Markdown("Selecciona los archivos y haz clic en 'Subir Archivos e Iniciar Procesamiento'")
    
    # Progress section (initially hidden)
    with gr.Group(visible=False) as progress_box:
        gr.Markdown("## 📊 Progreso del Procesamiento")
        progress_text = gr.Markdown("Iniciando...")
        progress_bar = gr.Slider(
            minimum=0, 
            maximum=100, 
            value=0, 
            label="Progreso (%)",
            interactive=False,
            show_label=True
        )
        
        # Manual refresh button
        refresh_btn = gr.Button("🔄 Actualizar Progreso", variant="secondary")
    
    # Logs section (initially hidden)
    with gr.Group(visible=False) as logs_box:
        gr.Markdown("## 📝 Logs del Procesamiento")
        logs_display = gr.Textbox(
            label="Logs en tiempo real",
            lines=10,
            max_lines=20,
            interactive=False,
            show_copy_button=True
        )
    
    # Download section (initially hidden)
    with gr.Group(visible=False) as download_box:
        gr.Markdown("## ✅ Procesamiento Completado")
        gr.Markdown("El procesamiento ha finalizado exitosamente. ¡Aquí están tus predicciones!")
        
        # Statistics section
        stats_display = gr.Markdown("", visible=True)
        
        # Results table
        gr.Markdown("### 🎯 Top 100 Predicciones (Mayor Probabilidad)")
        results_display = gr.Dataframe(
            label="Predicciones de Compra",
            headers=["Cliente ID", "Producto ID", "Semana", "Probabilidad"],
            datatype=["number", "number", "number", "number"],
            interactive=False,
            wrap=True
        )
        
        # Download button
        download_url_hidden = gr.State("")
        with gr.Row():
            download_btn = gr.Button("📥 Descargar CSV Completo", variant="secondary")
            gr.Markdown("💡 **Tip:** La tabla muestra las 100 predicciones con mayor probabilidad. El CSV completo contiene todas las recomendaciones.")
    
    # Event handlers
    upload_btn.click(
        fn=subir_archivos_y_iniciar,
        inputs=[clientes_file, productos_file, transacciones_file],
        outputs=[
            status_msg,
            progress_box,
            logs_box,  
            download_box,
            progress_text,
            progress_bar,
            logs_display,
            download_url_hidden,
            job_id_state,
            results_display,
            stats_display
        ]
    )
    
    refresh_btn.click(
        fn=actualizar_progreso,
        inputs=[job_id_state],
        outputs=[
            progress_box,
            logs_box,
            download_box,
            progress_text,
            progress_bar,
            logs_display,
            download_url_hidden,
            results_display,
            stats_display
        ]
    )
    
    download_btn.click(
        fn=lambda url: cargar_y_mostrar_resultados(url)[0],  # Solo retorna el DataFrame
        inputs=[download_url_hidden],
        outputs=[results_display]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
