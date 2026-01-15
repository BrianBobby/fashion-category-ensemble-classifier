import os
import sys
import io
import traceback
from PIL import Image
import imageio
import numpy as np
from flask import Flask, request, jsonify, render_template
import tensorflow as tf

# ---------------- CONFIG ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")   # Put your .keras files here
# If you prefer absolute paths, replace with e.g.
# MODEL_DIR = r"C:\Users\linub\OneDrive\Desktop\Project\models"

VGG_FILENAME = "deepfashion_vgg16_best_model_50epoch_no_dress.keras"
RESNET_FILENAME = "deepfashion_resnet50_final_model.keras"

VGG_PATH = os.path.join(MODEL_DIR, VGG_FILENAME)
RESNET_PATH = os.path.join(MODEL_DIR, RESNET_FILENAME)

IMG_SIZE = (224, 224)

CLASS_NAMES = [
    'Blazer', 'Blouse', 'Cardigan', 'Jacket', 'Jeans', 'Jumpsuit',
    'Romper', 'Shorts', 'Skirt', 'Sweater', 'Sweatpants', 'Tank', 'Tee', 'Top'
]

# ----------------- APP -------------------
app = Flask(__name__, static_folder="static", template_folder="templates")

# ---------- diagnostics before loading ----------
print("Base directory:", BASE_DIR)
print("Model directory:", MODEL_DIR)
print("Expecting model files:")
print(" - VGG :", VGG_PATH)
print(" - ResNet :", RESNET_PATH)

for p in (VGG_PATH, RESNET_PATH):
    if os.path.exists(p):
        print(f"FOUND: {p} ({os.path.getsize(p):,} bytes)")
    else:
        print(f"MISSING: {p}")

if not (os.path.exists(VGG_PATH) and os.path.exists(RESNET_PATH)):
    print("\nERROR: One or both model files missing. Put the .keras files in the `models/` folder or update MODEL_DIR.")
    sys.exit(1)

# -------------- load models ----------------
print("\nLoading models (this may take a while)...")
try:
    # Use compile=False to avoid issues if optimizer not available
    vgg_model = tf.keras.models.load_model(VGG_PATH, compile=False)
    resnet_model = tf.keras.models.load_model(RESNET_PATH, compile=False)
    print("Models loaded successfully.")
except Exception as e:
    print("Model loading failed. Full traceback:")
    traceback.print_exc()
    sys.exit(1)

# ---------- preprocessing helpers ------------
def preprocess_vgg_from_pil(pil_img):
    img = pil_img.resize(IMG_SIZE)
    arr = np.array(img).astype("float32")
    if arr.ndim == 2:  # grayscale
        arr = np.stack([arr]*3, axis=-1)
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    arr = np.expand_dims(arr, axis=0)
    arr = tf.keras.applications.vgg16.preprocess_input(arr)
    return arr

def preprocess_resnet_from_pil(pil_img):
    img = pil_img.resize(IMG_SIZE)
    arr = np.array(img).astype("float32")
    if arr.ndim == 2:
        arr = np.stack([arr]*3, axis=-1)
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    arr = np.expand_dims(arr, axis=0)
    arr = tf.keras.applications.resnet50.preprocess_input(arr)
    return arr

# ---------- ensemble logic (same as your Kaggle code) ----------
def advanced_ensemble(vgg_pred, resnet_pred):
    vgg_idx = int(np.argmax(vgg_pred))
    resnet_idx = int(np.argmax(resnet_pred))
    vgg_conf = float(vgg_pred[vgg_idx])
    resnet_conf = float(resnet_pred[resnet_idx])

    # If both agree
    if vgg_idx == resnet_idx:
        return vgg_idx

    # If VGG is very confident, choose it
    if vgg_conf >= 0.65:
        return vgg_idx

    # If ResNet is significantly more confident, choose it
    if resnet_conf > vgg_conf + 0.25:
        return resnet_idx

    # Weighted average favoring VGG
    combined_pred = 0.75 * vgg_pred + 0.25 * resnet_pred
    ensemble_idx = int(np.argmax(combined_pred))

    # If ensemble disagrees but VGG still somewhat confident, keep VGG
    if ensemble_idx != vgg_idx and vgg_conf > 0.4:
        return vgg_idx

    return ensemble_idx

# ---------------- routes --------------------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        if "file" not in request.files:
            return jsonify({"error": "no file part"}), 400

        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "no selected file"}), 400

        img_bytes = file.read()
        # Try Pillow first, fallback to imageio for unsupported formats (like some WebP builds)
        pil_img = None
        try:
            pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        except Exception as pil_err:
            # fallback
            try:
                arr = imageio.v2.imread(io.BytesIO(img_bytes))
                pil_img = Image.fromarray(arr).convert("RGB")
            except Exception as io_err:
                raise RuntimeError(f"Pillow error: {pil_err}; imageio error: {io_err}")

        # Preprocess and predict
        vgg_input = preprocess_vgg_from_pil(pil_img)
        resnet_input = preprocess_resnet_from_pil(pil_img)

        vgg_pred = vgg_model.predict(vgg_input, verbose=0)[0]
        resnet_pred = resnet_model.predict(resnet_input, verbose=0)[0]

        ensemble_idx = advanced_ensemble(vgg_pred, resnet_pred)
        combined_pred = 0.75 * vgg_pred + 0.25 * resnet_pred
        confidence = float(combined_pred[ensemble_idx])

        # Build top-k
        top_k = 5
        top_indices = combined_pred.argsort()[-top_k:][::-1]
        top_predictions = [
            {"class": CLASS_NAMES[int(i)], "prob": float(combined_pred[int(i)])}
            for i in top_indices
        ]

        return jsonify({
            "predicted_class": CLASS_NAMES[int(ensemble_idx)],
            "confidence": confidence,
            "top_predictions": top_predictions,
            "vgg_pred": [float(x) for x in vgg_pred],
            "resnet_pred": [float(x) for x in resnet_pred]
        }), 200

    except Exception as e:
        tb = traceback.format_exc()
        print("Prediction error:\n", tb)
        return jsonify({"error": "prediction_failed", "detail": str(e)}), 500

# -------------- run -------------------------
if __name__ == "__main__":
    # For local dev
    app.run(host="0.0.0.0", port=5000, debug=False)
