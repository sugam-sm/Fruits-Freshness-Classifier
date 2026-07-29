from flask import Flask, request, jsonify, render_template
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn.functional as F
from PIL import Image
import io

app = Flask(__name__)

# --------------------------------------------------
# Device setup
# --------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------------------------------
# Class names
# IMPORTANT:
# This order must match train_dataset.classes from training.
# From your notebook, the class order appears to be:
# ['freshapples', 'freshbanana', 'freshoranges',
#  'rottenapples', 'rottenbanana', 'rottenoranges']
# --------------------------------------------------
class_names = [
    "freshapples",
    "freshbanana",
    "freshoranges",
    "rottenapples",
    "rottenbanana",
    "rottenoranges"
]


# --------------------------------------------------
# Model 1: Baseline CNN
# Must match training architecture exactly
# --------------------------------------------------
class BaselineCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 56 * 56, 6)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# --------------------------------------------------
# Model 2: Improved CNN
# Must match your final notebook architecture exactly
# --------------------------------------------------
class ImprovedCNN(nn.Module):
    def __init__(self, dropout_rate=0.3):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 6)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = self.classifier(x)
        return x


# --------------------------------------------------
# Model 3: ResNet18 Transfer Learning Model
# Must match training architecture exactly
# --------------------------------------------------
def get_resnet_model():
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 6)
    return model


# --------------------------------------------------
# Load saved model weights
# --------------------------------------------------
model1 = BaselineCNN()
model1.load_state_dict(torch.load("baseline_best.pth", map_location=device))
model1.to(device)
model1.eval()

model2 = ImprovedCNN(dropout_rate=0.3)
model2.load_state_dict(torch.load("improved_best.pth", map_location=device))
model2.to(device)
model2.eval()

model3 = get_resnet_model()
model3.load_state_dict(torch.load("transfer_best.pth", map_location=device))
model3.to(device)
model3.eval()


# --------------------------------------------------
# Image transformation for inference
# Must match validation/test transform from training
# --------------------------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# --------------------------------------------------
# Prediction function
# --------------------------------------------------
def predict(model, image_tensor):
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = F.softmax(output[0], dim=0)

        predicted_index = torch.argmax(probabilities).item()
        confidence = probabilities[predicted_index].item()

        predicted_class = class_names[predicted_index]

        return predicted_class, confidence


# --------------------------------------------------
# Home route
# --------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html")


# --------------------------------------------------
# Prediction route
# --------------------------------------------------
@app.route("/predict", methods=["POST"])
def predict_route():
    if "image" not in request.files:
        return jsonify({"error": "No image file uploaded"}), 400

    file = request.files["image"]

    if file.filename == "":
        return jsonify({"error": "No selected image"}), 400

    image = Image.open(io.BytesIO(file.read())).convert("RGB")

    image_tensor = transform(image).unsqueeze(0).to(device)

    result1, conf1 = predict(model1, image_tensor)
    result2, conf2 = predict(model2, image_tensor)
    result3, conf3 = predict(model3, image_tensor)

    return jsonify({
        "model1": f"{result1} ({conf1 * 100:.2f}%)",
        "model2": f"{result2} ({conf2 * 100:.2f}%)",
        "model3": f"{result3} ({conf3 * 100:.2f}%)"
    })


# --------------------------------------------------
# Run Flask app
# --------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True)