from flask import Flask, render_template, request, jsonify
import numpy as np
from PIL import Image
from io import BytesIO
import tensorflow as tf
import os

app = Flask(__name__)

MODEL = tf.keras.models.load_model("../models/2")
CLASS_NAME = ['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']


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

        # Return the prediction result
        return predicted_class, confidence
    except Exception as e:
        return "Error", str(e)


@app.route("/", methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        try:
            # Ensure 'image' is the correct name for the file input field
            file = request.files['image']
            filename = file.filename
            print("@@ Input posted = ", filename)

            file_path = os.path.join('api/static/user_uploaded', filename)
            file.save(file_path)
            if file:

                # Get the prediction result without saving the file
                predicted_class, confidence = predict_disease(file.read())

                # Render different HTML templates based on the prediction
                if predicted_class == 'Potato___Early_blight':
                    return render_template('Early_blight.html')
                elif predicted_class == 'Potato___Late_blight':
                    return render_template('Late_Blight.html', predicted_class=predicted_class, confidence=confidence, user_image=file_path)
                elif predicted_class == 'Potato___healthy':
                    return render_template('healthy_plant.html', predicted_class=predicted_class, confidence=confidence)
                else:
                    return render_template('unknown.html', predicted_class=predicted_class, confidence=confidence)
            else:
                return jsonify({
                    "error": "No file provided"
                })
        except Exception as e:
            return jsonify({
                "error": str(e)
            })
    else:
        return render_template('index.html')


if __name__ == "__main__":
    app.run(threaded=False)
