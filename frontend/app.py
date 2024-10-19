from flask import Flask, request, jsonify, render_template
import tensorflow as tf
from chat import chatbot
from PIL import Image
import numpy as np
import io

app = Flask(__name__)

model = tf.keras.applications.MobileNetV2(weights="imagenet")

def preprocess_image(image):
    image = image.resize((224, 224))
    image = np.array(image)
    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
    return np.expand_dims(image, axis=0)

def generate_response(detected_objects):
    if detected_objects:
        response = f"I detected the following objects in the image: {', '.join(detected_objects)}."
    else:
        response = "I couldn't detect any objects in the image."
    return response

@app.route('/')
def index():
    return render_template('index.html')

# Chatbot route
@app.route('/ask', methods=['POST'])
def ask():
    data = request.get_json()
    message = data.get("message", "")
    bot_response = chatbot(message)
    return jsonify({"answer": bot_response})

# Image upload route
@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400

    image = request.files['image'].read()
    image = Image.open(io.BytesIO(image))

    processed_image = preprocess_image(image)

    predictions = model.predict(processed_image)
    decoded_predictions = tf.keras.applications.mobilenet_v2.decode_predictions(predictions, top=3)[0]
    detected_objects = [label for (_, label, _) in decoded_predictions]

    response_text = generate_response(detected_objects)

    return jsonify({"detected_objects": detected_objects, "response": response_text})

if __name__ == '__main__':
    app.run(debug=True)
