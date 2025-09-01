from flask import Flask, request, jsonify, send_file, render_template
import torch
import os
import uuid
import cv2
from ultralytics import YOLO
from werkzeug.utils import secure_filename
from flask_cors import CORS
import sys
import traceback

app = Flask(__name__)
CORS(app)

# =================== PATH TO LOCAL YOLOv5 ===================
sys.path.insert(0, r"D:/YOLOV5")

# =================== LOAD MODELS ===================
print("Loading YOLO models...")

# --- YOLOv5 ---
model_v5 = torch.hub.load(
    'D:/YOLOV8/yolov5',
    'custom',
    path=r"D:\YOLOV8\yolov5\runs\train\exp\weights\best_windows.pt",
    source='local',
    force_reload=True
)


print("YOLOv5 loaded from local.")

# --- YOLOv8 ---
model_v8 = YOLO(r"D:/YOLOV8/runs/detect/train/weights/best.pt")
print("YOLOv8 loaded.")

# =================== FOLDERS ===================
UPLOAD_FOLDER = "uploads"
RESULT_FOLDER = "results"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# =================== HOME ===================
@app.route("/")
def home():
    return render_template("index.html")

# =================== HELPER ===================
def save_result_image(img_array, file_id):
    output_path = os.path.join(RESULT_FOLDER, f"{file_id}.jpg")
    cv2.imwrite(output_path, img_array)
    return output_path

# =================== YOLOv5 API ===================
# YOLOv5 API
@app.route("/predict_v5", methods=["POST"])
def predict_v5():
    print("Received /predict_v5 request")
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files["file"]
        file_id = str(uuid.uuid4())
        filename = secure_filename(file.filename)
        input_path = os.path.join(UPLOAD_FOLDER, f"{file_id}_{filename}")
        file.save(input_path)
        print("YOLOv5 - File saved:", input_path)

        # Run inference
        results = model_v5(input_path)          
        annotated = results.render()[0]

        # ✅ Fix màu cho YOLOv5
        annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

        output_path = save_result_image(annotated, file_id)
        print("YOLOv5 - Result saved:", output_path)

        return jsonify({"result_url": f"/results/{os.path.basename(output_path)}"})

    except Exception as e:
        print("YOLOv5 error:", e)
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# =================== YOLOv8 API ===================
@app.route("/predict_v8", methods=["POST"])
def predict_v8():
    print("Received /predict_v8 request")
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files["file"]
        file_id = str(uuid.uuid4())
        filename = secure_filename(file.filename)
        input_path = os.path.join(UPLOAD_FOLDER, f"{file_id}_{filename}")
        file.save(input_path)
        print("YOLOv8 - File saved:", input_path)

        # Run inference
        results = model_v8(input_path)
        annotated = results[0].plot()  # YOLOv8 trả về RGB nhưng OpenCV vẫn save chuẩn
        output_path = save_result_image(annotated, file_id)
        print("YOLOv8 - Result saved:", output_path)

        return jsonify({"result_url": f"/results/{os.path.basename(output_path)}"})

    except Exception as e:
        print("YOLOv8 error:", e)
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
    app.run(debug=True)
