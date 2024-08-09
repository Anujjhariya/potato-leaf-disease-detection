import numpy as np
import uvicorn
from flask import Flask,render_template,request
from fastapi import FastAPI,UploadFile,File
import uvicorn
from io import BytesIO
import numpy as np
from PIL import Image
import tensorflow as tf
import asyncio

# app = FastAPI()
app = Flask(__name__)

MODEL = tf.keras.models.load_model("../models/2")
CLASS_NAME = ['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']


@app.route("/", methods=['GET', 'POST'])
def home():
        return render_template('index.html')

# @app.get("/ping")
# async def ping():
#     return "hello I am here"

def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image

# @app.post("/predict")
@app.route("/predict", methods = ['GET','POST'])
async def predict(
        # file: UploadFile = File(...)
        file = request.files['image']
):
    bytes = read_file_as_image(await file.read())
    image_batch = np.expand_dims(bytes,0)
    predictions = MODEL.predict(image_batch)
    predicted_class = CLASS_NAME[np.argmax(predictions)]
    confidence = np.max(predictions[0])
    print(predicted_class,confidence)
    return {
        "class":predicted_class,
        "confidence": float(confidence)
    }


if __name__ == "__main__":
    app.run(threaded = False,)
    # uvicorn.run(app,host='localhost',port=8085)