import os
from flask import Flask, request, jsonify
from model import EmotionModel  # import your Hugging Face model

app = Flask(__name__)

# Lazy load
emotion_model = None

def get_emotion_model():
    global emotion_model
    if emotion_model is None:
        emotion_model = EmotionModel()  # load on first request
        print("EmotionModel loaded!")
    return emotion_model

# -----------------------
# Routes
# -----------------------
@app.route('/')
def home():
    # Serve your front-end HTML page
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    data = request.files.get("image")  # Expecting an image file
    if data is None:
        return jsonify({"error": "No image provided"}), 400
    from PIL import Image

    pil_image = Image.open(data)
    pil_image = pil_image.resize((224, 224))

    model_instance = get_emotion_model()  # lazy load
    prediction = model_instance.predict(pil_image)

    return jsonify(prediction)

# -----------------------
# Run app on Render port
# -----------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
