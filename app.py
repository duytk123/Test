from flask import Flask, request, jsonify, send_file, render_template
import torch
import os
import uuid
import cv2
from ultralytics import YOLO
from werkzeug.utils import secure_filename
from flask_cors import CORS
import traceback

app = Flask(__name__)
CORS(app)

# =================== LOAD MODELS ===================
print("Loading YOLO models...")

# --- YOLOv5 ---
model_v5 = torch.hub.load(
    './yolov5',  # repo YOLOv5 nằm trong thư mục project
    'custom',
    path='./runs/train/exp/weights/best_windows.pt',  # chỉnh đường dẫn cho phù hợp
    source='local'
)
print("YOLOv5 loaded.")

# --- YOLOv8 ---
model_v8 = YOLO('./runs/detect/train/weights/best.pt')
print("YOLOv8 loaded.")

# =================== FOLDERS ===================
UPLOAD_FOLDER = "uploads"
RESULT_FOLDER = "results"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

@app.route("/")
def home():
    return render_template("index.html")

def save_result_image(img_array, file_id):
    output_path = os.path.join(RESULT_FOLDER, f"{file_id}.jpg")
    cv2.imwrite(output_path, img_array)
    return output_path

# =================== YOLOv5 API ===================
@app.route("/predict_v5", methods=["POST"])
def predict_v5():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files["file"]
        file_id = str(uuid.uuid4())
        filename = secure_filename(file.filename)
        input_path = os.path.join(UPLOAD_FOLDER, f"{file_id}_{filename}")
        file.save(input_path)

        results = model_v5(input_path)
        annotated = results.render()[0]
        annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

        output_path = save_result_image(annotated, file_id)

        return jsonify({"result_url": f"/results/{os.path.basename(output_path)}"})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

# =================== YOLOv8 API ===================
@app.route("/predict_v8", methods=["POST"])
def predict_v8():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files["file"]
        file_id = str(uuid.uuid4())
        filename = secure_filename(file.filename)
        input_path = os.path.join(UPLOAD_FOLDER, f"{file_id}_{filename}")
        file.save(input_path)

        results = model_v8(input_path)
        annotated = results[0].plot()
        output_path = save_result_image(annotated, file_id)

        return jsonify({"result_url": f"/results/{os.path.basename(output_path)}"})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

# =================== SERVE RESULTS ===================
@app.route("/results/<path:filename>")
def serve_result(filename):
    file_path = os.path.join(RESULT_FOLDER, filename)
    if os.path.exists(file_path):
        return send_file(file_path)
    return jsonify({"error": "File not found"}), 404

# =================== RUN APP ===================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
