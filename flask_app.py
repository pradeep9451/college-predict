import os
import gdown
import joblib
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

MODEL_PATH = "trained_model.joblib"
GDRIVE_URL = "https://drive.google.com/uc?id=1Zw_9DvCyTOPC7jshwCqEhWNru4iTmytN"

# 🧠 Lazy load model only during predict
def load_model():
    if not os.path.exists(MODEL_PATH):
        print("📥 Downloading model...")
        gdown.download(GDRIVE_URL, MODEL_PATH, quiet=False)
    return joblib.load(MODEL_PATH)

@app.route("/")
def home():
    return "✅ College Predictor API is running"

@app.route("/options", methods=["GET"])
def get_options():
    model = load_model()
    colleges = sorted(set(k.split('|')[0] for k in model.keys()))
    branches = sorted(set(k.split('|')[1] for k in model.keys()))
    categories = sorted(set(k.split('|')[2] for k in model.keys()))
    return jsonify({
        "colleges": colleges,
        "branches": branches,
        "categories": categories
    })

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    college = data.get("college")
    branch = data.get("branch")
    category = data.get("category")
    year = data.get("year", 2024)

    if not (college and branch and category):
        return jsonify({"error": "Missing input fields"}), 400

    model = load_model()  # 🔄 Load here

    key = f"{college}|{branch}|{category}"
    if key not in model:
        return jsonify({"error": "No data found for that combination"}), 404

    try:
        predicted = model[key].predict([[year]])[0]
        return jsonify({"predicted_cutoff": round(float(predicted), 2)})
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # ✅ required by Render
    app.run(host="0.0.0.0", port=port)
