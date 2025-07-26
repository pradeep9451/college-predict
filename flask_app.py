from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib

app = Flask(__name__)
CORS(app)

# Load model dictionary
model = joblib.load("trained_model.joblib")

# Root route for browser check
@app.route("/", methods=["GET"])
def home():
    return "✅ College Predictor API is running. Use /predict for POST request."

# 🔍 Get available options for dropdowns
@app.route("/options", methods=["GET"])
def get_options():
    colleges = sorted(set(k.split('|')[0] for k in model.keys()))
    branches = sorted(set(k.split('|')[1] for k in model.keys()))
    categories = sorted(set(k.split('|')[2] for k in model.keys()))
    return jsonify({
        "colleges": colleges,
        "branches": branches,
        "categories": categories
    })

# 🎯 Predict route
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    college = data.get("college")
    branch = data.get("branch")
    category = data.get("category")
    year = data.get("year", 2024)

    # 🔐 Input validation
    if not (college and branch and category):
        return jsonify({"error": "Missing input fields"}), 400

    key = f"{college}|{branch}|{category}"

    if key not in model:
        return jsonify({"error": "No data found for that combination"}), 404

    try:
        predicted = model[key].predict([[year]])[0]
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

    return jsonify({"predicted_cutoff": round(float(predicted), 2)})

if __name__ == "__main__":
    app.run(debug=True)
