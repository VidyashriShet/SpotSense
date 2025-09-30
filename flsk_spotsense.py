from flask import Flask, request, jsonify, render_template, url_for
from flask_cors import CORS
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os
import json
import torch.nn.functional as F
from googletrans import Translator
from dotenv import load_dotenv
from werkzeug.utils import secure_filename


load_dotenv()
ORS_API_KEY = os.getenv("ORS_API_KEY")

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Model setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset_path = "spotsense_temples_preprocessed_dataset"
num_classes = len(os.listdir(dataset_path))

model = models.resnet50(pretrained=False)
model.fc = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(model.fc.in_features, num_classes)
)
model.load_state_dict(torch.load("spotsense_updated_30_resnet50_temples_model.pth", map_location=device))
model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

with open("temple_info.json", "r", encoding="utf-8") as file:
    temple_info = json.load(file)

class_names = os.listdir(dataset_path)
translator = Translator()

def predict_image(image):
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(image)
        probabilities = F.softmax(output, dim=1)
        confidence, predicted_class = torch.max(probabilities, 1)

    if confidence.item() < 0.5:
        return "Information on this place currently unavailable"

    return class_names[predicted_class.item()]

@app.route("/")
def home():
    return render_template("index.html", ors_api_key=ORS_API_KEY)


@app.route("/predict", methods=["POST"])
def predict():
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}

    def allowed_file(filename):
        return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

    if "file" not in request.files:
        return render_template("index.html", error="No file uploaded", ors_api_key=ORS_API_KEY)

    file = request.files["file"]
    if file.filename == "":
        return render_template("index.html", error="No selected file", ors_api_key=ORS_API_KEY)

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(file_path)

        try:
            image = Image.open(file_path).convert("RGB")
        except Exception:
            os.remove(file_path)
            return render_template("index.html", error="Invalid image file. Please upload a valid image.",
                                   ors_api_key=ORS_API_KEY)

        predicted_temple = predict_image(image)

        return render_template("index.html",
                               image_url=url_for('static', filename=f'uploads/{filename}'),
                               prediction=predicted_temple,
                               ors_api_key=ORS_API_KEY)
    else:
        return render_template("index.html", error="Please upload image files only (jpg, jpeg, png, webp).",
                               ors_api_key=ORS_API_KEY)


@app.route("/get_info/<temple_name>/<lang>")
def get_info(temple_name, lang):
    if temple_name in temple_info:
        info = temple_info[temple_name]
        if lang == "kn":
            info = {k: translator.translate(v, src="en", dest="kn").text for k, v in info.items()}
        return jsonify(info)
    return jsonify({"error": "No information available"})


@app.route('/social')
def social():
    return render_template('social.html')

if __name__ == "__main__":
    app.run(debug=True)
