import gradio as gr
import requests

def subir_y_predecir(clientes, productos, transacciones):
    files = {
        "clientes": clientes,
        "productos": productos,
        "transacciones": transacciones
    }
    response = requests.post("http://backend:8000/upload_and_predict", files=files)
    if response.status_code == 200 and response.headers.get("content-type") == "text/csv":
        return response.text
    return response.json()

demo = gr.Interface(
    fn=subir_y_predecir,
    inputs=[
        gr.File(label="Clientes.parquet"),
        gr.File(label="Productos.parquet"),
        gr.File(label="Transacciones.parquet"),
    ],
    outputs="text",
    title="SodAI Drinks 🥤",
    description="Sube los archivos para obtener las predicciones semanales de compra."
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)