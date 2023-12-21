from fastapi import FastAPI
from fastapi import UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from starlette.middleware.cors import CORSMiddleware
from transformers import pipeline
import os
import shutil
import tempfile



app = FastAPI()


# Configurar el middleware de CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Esto permite cualquier origen
    allow_credentials=True,
    allow_methods=["*"],   # Esto permite todos los métodos HTTP
    allow_headers=["*"],   # Esto permite todos los encabezados HTTP
)


@app.post("/clasificarImagen")
async def clasificarImagen(image: UploadFile = File(...)):
    if not (image.filename.endswith(".svg") or image.filename.endswith(".png") or image.filename.endswith(".jpeg") or image.filename.endswith(".jpg")):
        raise HTTPException(status_code=400, detail="El archivo debe ser una imagen.")
    try:
        # Crear un archivo temporal para guardar la imagen
        with tempfile.NamedTemporaryFile(delete=False) as temp_image:
            shutil.copyfileobj(image.file, temp_image)
            temp_image_path = temp_image.name  # Obtener el path temporal de la imagen

        resultado = obtenerClasificacion(temp_image_path)  # Llamar a la función con el path temporal
        os.remove(temp_image_path)          # Eliminar el archivo temporal después de usarlo
        return resultado
    except Exception as e:
        return HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    # Ejecutar la aplicación en host="0.0.0.0" y port=8000
    uvicorn.run(app, host="0.0.0.0", port=8000)

def obtenerClasificacion(path):
    pipe = pipeline("image-classification", model="Giecom/giecom-vit-model-clasification-waste")

    prediction = pipe(path)

    class_label = prediction[0]['label']
    probability = prediction[0]['score']

    return {"label":class_label,"probability":probability}
