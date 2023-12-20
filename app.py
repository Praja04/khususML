import os

import numpy as np
from flask import Flask, jsonify, request
from PIL import Image
from tensorflow.keras.models import load_model
from werkzeug.utils import secure_filename
from auth import auth

app = Flask(__name__)
app.config['ALLOWED_EXTENSIONS'] = set(['png', 'jpg', 'jpeg'])
app.config['UPLOAD_FOLDER'] = 'static/uploads/'
app.config['MODEL_FILE'] = 'Biscuits_lstm_model.h5'


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in app.config['ALLOWED_EXTENSIONS']


model = load_model(app.config['MODEL_FILE'], compile=False)


def predict_image_category(image):
    try:
        # Proses gambar
        img = Image.open(image).convert("RGB")
        img = img.resize((224, 224))
        img_array = np.asarray(img)
        img_array = np.expand_dims(img_array, axis=0)
        normalized_image_array = (img_array.astype(np.float32) / 127.5) - 1
        data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
        data[0] = normalized_image_array
        predictions = model.predict(data)
        index = np.argmax(predictions)
        class_name = labels[index]
        confidence_score = predictions[0][index]
        return class_name[2:], confidence_score
    except Exception as e:
        raise ValueError(f"Error processing image: {e}")

@app.route("/prediction", methods=["POST"])
@auth.login_required()
def prediction_route():
    if request.method == "POST":
        image = request.files.get("image")
        text = request.form.get("text")

        if image and allowed_file(image.filename):
            # Proses gambar
            filename = secure_filename(image.filename)
            image.save(os.path.join(app.config["UPLOAD_FOLDER"], filename))
            image_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)

            class_name, confidence_score = predict_image_category(image_path)
        elif text:
            # Proses teks
            class_name, confidence_score = predict_text(text)
        else:
            return jsonify({
                "status": {
                    "code": 400,
                    "message": "Invalid input. Please provide either an image or text."
                },
                "data": None,
            }), 400

        return jsonify({
            "status": {
                "code": 200,
                "message": "Success predicting"
            },
            "data": {
                "prediction": class_name,
                "confidence": float(confidence_score),
            }
        }), 200
    else:
        return jsonify({
            "status": {
                "code": 405,
                "message": "Method not allowed"
            },
            "data": None,
        }), 405


if __name__ == "__main__":
    app.run(debug=True,
            host="0.0.0.0",
            port=int(os.environ.get("PORT", 8080)))