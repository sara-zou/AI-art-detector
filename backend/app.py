from flask import Flask, request, jsonify
from PIL import UnidentifiedImageError
from flask_cors import CORS
import config
from helpers import get_device, load_model, predict


app = Flask(__name__)
CORS(app)
device = get_device()
model = load_model(device)
print(f"[Artifex] Device: {device}")

@app.get("/")
def health():
    return jsonify({"status": "ok", "device": str(device)})

@app.post("/api/predict")
def api_predict():
    #validate appropraite file existence
    if "image" not in request.files:
        return jsonify({"error": "No image field in request"}), 400
    
    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400
    
    #validate file
    if file.mimetype not in config.ALLOWED_MIME_TYPES:
        return jsonify({
            "error": f"Unsupported file type '{file.mimetype}'. "
                     f"Accepted: jpeg, png, webp, bmp"
        }), 415
    
    #predict file
    file_bytes = file.read()
    result = predict(file_bytes, model, device)
    
    return jsonify(result)

@app.errorhandler(413)
def request_too_large(e):
    return jsonify({"error": f"File too large"}), 413
 
@app.errorhandler(405)
def method_not_allowed(e):
    return jsonify({"error": "Method not allowed"}), 405

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=config.PORT)