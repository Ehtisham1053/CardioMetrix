from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from .predictor import predict

app = Flask(__name__, template_folder="templates", static_folder="static")
CORS(app)

@app.get("/health")
def health():
    return jsonify({"status":"ok"}), 200

@app.get("/")
def index():
    return render_template("index.html")

@app.post("/predict")
def predict_endpoint():
    try:
        payload = request.get_json(force=True) or {}
    except Exception:
        return jsonify({"error":"Invalid JSON payload"}), 400

    if "age" not in payload:
        return jsonify({"error":"Missing required field: age"}), 400

    try:
        out = predict(payload)
        out["disclaimer"] = (
            "Educational decision support. Not a diagnosis. "
            "Use clinical judgment and confirm with appropriate tests."
        )
        return jsonify(out), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
