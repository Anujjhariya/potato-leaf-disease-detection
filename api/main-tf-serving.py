from fastapi import FastAPI, UploadFile, File
import uvicorn
from io import BytesIO
import numpy as np
from PIL import Image
import tensorflow as tf
import requests

app = FastAPI()

endpoint = "http://localhost:8085/models/predict"
CLASS_NAME = ['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']


@app.get("/ping")
async def ping():
    return "hello I am here"

def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image


@app.post("/predict")
async def predict(
        file: UploadFile = File(...)
):
    image = read_file_as_image(await file.read())
    image_batch = np.expand_dims(image, 0)
    json_data = {
        "instances": image_batch.tolist()
    }
    response = requests.post(endpoint, json=json_data)
    pass

    #
    #
    # predictions = MODEL.predict(image_batch)
    # predicted_class = CLASS_NAME[np.argmax(predictions)]
    # confidence = np.max(predictions[0])
    # print(predicted_class,confidence)
    # return {
    #     "class": predicted_class,
    #     "confidence": float(confidence)
    # }


if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8085)
