import os
from flask import Flask, request, render_template
from model import EmotionModel  # your pretrained Hugging Face model
from PIL import Image

app = Flask(__name__)

# -------------------
# Lazy load model
# -------------------
emotion_model = None
def get_emotion_model():
    global emotion_model
    if emotion_model is None:
        emotion_model = EmotionModel()  # load on first request
        print("EmotionModel loaded!")
    return emotion_model

# -------------------
# Routes
# -------------------
@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files.get("image")
    if file is None:
        return render_template("index.html", prediction="No image provided")

    # Preprocess image
    pil_image = Image.open(file).resize((224, 224))

    # Predict
    model_instance = get_emotion_model()
    prediction = model_instance.predict(pil_image)

    # Render result on the same page
    return render_template("index.html", prediction=prediction)

# -------------------
# Run on Render port
# -------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
