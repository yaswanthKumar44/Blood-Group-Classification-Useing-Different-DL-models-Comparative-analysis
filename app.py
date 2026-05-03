import os
import torch
import timm
import numpy as np
from PIL import Image
from flask import Flask, render_template, request
import albumentations as A
from albumentations.pytorch import ToTensorV2

# ===============================
# FLASK APP SETUP
# ===============================
app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

device = "cuda" if torch.cuda.is_available() else "cpu"

# ===============================
# CLASS LABELS (MUST MATCH TRAINING)
# ===============================
classes = ["A+", "A-", "AB+", "AB-", "B+", "B-", "O+", "O-"]  
# ⚠️ adjust if your dataset order is different

# ===============================
# LOAD MODEL
# ===============================
model = timm.create_model(
    "tf_efficientnetv2_s",
    pretrained=False,
    num_classes=len(classes)
)

model.load_state_dict(
    torch.load("C:/Users/yashy/OneDrive - Lakireddy Bali Reddy College of Engineering/Blood Group detection for comparative study paper/Efficientnet_V2s_98.14/Best_EfficientNetV2S.pth", map_location=device)
)
model.to(device)
model.eval()

print("✅ Model Loaded Successfully")

# ===============================
# IMAGE TRANSFORM
# ===============================
transform = A.Compose([
    A.Resize(224, 224),
    A.Normalize(),
    ToTensorV2()
])

# ===============================
# ROUTES
# ===============================
@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    image_path = None

    if request.method == "POST":
        file = request.files["image"]

        if file:
            image_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(image_path)

            img = np.array(Image.open(image_path).convert("RGB"))
            img = transform(image=img)["image"].unsqueeze(0).to(device)

            with torch.no_grad():
                output = model(img)
                probs = torch.softmax(output, dim=1)
                pred_idx = torch.argmax(probs, dim=1).item()
                confidence = probs[0][pred_idx].item() * 100

            prediction = f"{classes[pred_idx]} ({confidence:.2f}%)"

    return render_template(
        "home.html",
        prediction=prediction,
        image_path=image_path
    )

# ===============================
# RUN APP
# ===============================
if __name__ == "__main__":
    app.run(debug=True)
