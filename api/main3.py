from flask import Flask, render_template, request, jsonify
import numpy as np
from PIL import Image
from io import BytesIO
import tensorflow as tf

app = Flask(__name__)

MODEL = tf.keras.models.load_model("../models/2")
CLASS_NAME = ['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']


def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image


@app.route("/", methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        try:
            # Ensure 'image' is the correct name for the file input field
            file = request.files['image']

            if file:
                # Read the file and process the prediction
                bytes = read_file_as_image(file.read())
                image_batch = np.expand_dims(bytes, 0)
                predictions = MODEL.predict(image_batch)
                predicted_class = CLASS_NAME[np.argmax(predictions)]
                confidence = float(np.max(predictions[0]))
                if predicted_class == CLASS_NAME[0]:
                    return "Early Bligth Plant", 'Early_blight.html'  # if index 0 burned leaf
                elif predicted_class == CLASS_NAME[1]:
                    return 'Diseased Cotton Plant', 'Late_Blight.html'  # # if index 1
                elif predicted_class == CLASS_NAME[2]:
                    return 'Healthy Cotton Plant', 'api/templates/healthy_plant.html'  # if index 2  fresh leaf
                else:
                    return "Healthy Cotton Plant", 'api/templates/healthy_plant.html'  # if index 3
                # Print the result for debugging
                print(predicted_class, confidence)

                return jsonify({
                    "class": predicted_class,
                    "confidence": confidence
                })
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
