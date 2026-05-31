from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import os
import gdown

MODEL_PATH = "trained_model.joblib"

# Download model if not present
if not os.path.exists(MODEL_PATH):
    file_id = "13itJT7GpgA3wFi5z5dY5GLuT2yc0gYpa"
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, MODEL_PATH, quiet=False)

app = Flask(__name__)
CORS(app, origins=["http://localhost:3000"])

# Load trained model
try:
    model = joblib.load(MODEL_PATH)
    print("📦 Model loaded successfully")
except Exception as e:
    print(f"❌ Failed to load model: {e}")
    model = {}

@app.route("/", methods=["GET"])
def home():
    return "✅ College Predictor API is running. Use /predict for POST request."

@app.route("/options", methods=["GET"])
def get_options():
    branches = sorted(set(k.split('|')[1] for k in model.keys()))
    categories = sorted(set(k.split('|')[2] for k in model.keys()))
    return jsonify({"branches": branches, "categories": categories})

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    percentile_raw = data.get("percentile", "")
    try:
        percentile = float(percentile_raw)
    except (ValueError, TypeError):
        return jsonify({"error": "Percentile must be a number"}), 400

    branch = data.get("branch", "").strip()
    category = data.get("category", "").strip()

    if not branch or not category:
        return jsonify({"error": "Branch and category are required"}), 400

    year_raw = data.get("year", 2025)
    try:
        year = int(year_raw)
    except (ValueError, TypeError):
        year = 2025

    results = []
    for key, mdl in model.items():
        college, b, cat = key.split("|")
        if b.lower() == branch.lower() and cat.lower() == category.lower():
            try:
                cutoff = mdl.predict([[year]])[0]
                if percentile >= cutoff:
                    results.append({
                        "college": college,
                        "predicted_cutoff": round(float(cutoff), 2)
                    })
            except Exception as e:
                print(f"⚠️ Prediction failed for {key}: {e}")
                continue

    results = sorted(results, key=lambda x: x["predicted_cutoff"], reverse=True)[:20]

    if not results:
        return jsonify({"error": "No matching colleges found"}), 404

    return jsonify({"matches": results})

if __name__ == "__main__":
    app.run(debug=True)
