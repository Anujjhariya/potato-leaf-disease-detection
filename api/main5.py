from flask import Flask, render_template, request, jsonify
import numpy as np
from PIL import Image
from io import BytesIO
import tensorflow as tf
import os

app = Flask(__name__)

MODEL = tf.keras.models.load_model("../models/2")
CLASS_NAME = ['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']

# Create the 'static/user_uploaded' directory if it doesn't exist
if not os.path.exists('static/user_uploaded'):
    os.makedirs('static/user_uploaded')

def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image

def predict_disease(file_data):
    try:
        bytes = read_file_as_image(file_data)
        image_batch = np.expand_dims(bytes, 0)
        predictions = MODEL.predict(image_batch)
        predicted_class = CLASS_NAME[np.argmax(predictions)]
        confidence = float(np.max(predictions[0]))

        # Save the uploaded image
        file_path = f'static/user_uploaded/{np.random.randint(100000)}.png'
        Image.fromarray(bytes).save(file_path)

        # Return the prediction result and file path
        return predicted_class, confidence, file_path
    except Exception as e:
        return "Error", str(e), None

@app.route("/", methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        try:
            file = request.files['image']
            filename = file.filename
            print("@@ Input posted = ", filename)

            # Get the prediction result and file path without saving the file
            predicted_class, confidence, file_path = predict_disease(file.read())

            if predicted_class == 'Potato___Early_blight':
                return render_template('Early_blight.html',predicted_class=predicted_class, confidence=confidence, user_image=file_path)
            elif predicted_class == 'Potato___Late_blight':
                return render_template('Late_Blight.html', predicted_class=predicted_class, confidence=confidence, user_image=file_path)
            elif predicted_class == 'Potato___healthy':
                return render_template('healthy_plant.html', predicted_class=predicted_class, confidence=confidence, user_image=file_path)
            else:
                return render_template('unknown.html', predicted_class=predicted_class, confidence=confidence, user_image=file_path)

        except Exception as e:
            return jsonify({"error": str(e)})

    else:
        return render_template('index.html')

if __name__ == "__main__":
    app.run(threaded=False)
