from flask import Flask, render_template, request, url_for
from transformers import pipeline
from PIL import Image
import io, os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# Create upload folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load pretrained emotion detection model
emotion_model = pipeline("image-classification", model="mo-thecreator/vit-Facial-Expression-Recognition")


@app.route('/', methods=['GET', 'POST'])
def home():
    prediction_label = None
    image_path = None

    if request.method == 'POST':
        if 'image' not in request.files:
            return render_template('index.html', prediction=None, image_path=None)

        file = request.files['image']
        if file.filename == '':
            return render_template('index.html', prediction=None, image_path=None)

        # Save uploaded image
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(image_path)

        # Predict emotion
        image = Image.open(image_path)
        preds = emotion_model(image)
        prediction_label = preds[0]['label']

        # Pass relative path to template
        image_path = url_for('static', filename=f'uploads/{file.filename}')

    return render_template('index.html', prediction=prediction_label, image_path=image_path)


if __name__ == '__main__':
    app.run(debug=True)
