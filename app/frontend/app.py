import gradio as gr
import requests
import pandas as pd
from io import StringIO

def subir_y_predecir(clientes, productos, transacciones):
    files = {
        "clientes": clientes,
        "productos": productos,
        "transacciones": transacciones
    }
    print("Haciendo POST a http://backend:8000/upload_and_predict")
    response = requests.post("http://backend:8000/upload_and_predict", files=files)
    print(f"Respuesta del backend: {response.status_code} - {response.text[:100]}...")
    if response.status_code == 200 and response.headers.get("content-type").startswith("text/csv"):
        csv_text = response.text
        df = pd.read_csv(StringIO(csv_text))
        return df
    return response.json()

demo = gr.Interface(
    fn=subir_y_predecir,
    inputs=[
        gr.File(label="Clientes.parquet"),
        gr.File(label="Productos.parquet"),
        gr.File(label="Transacciones.parquet"),
    ],
    outputs=gr.Dataframe(type="pandas", interactive=True),
    title="SodAI Drinks 🥤",
    description="Sube los archivos para obtener las predicciones semanales de compra."
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
