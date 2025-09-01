// ================== GLOBAL VARIABLES ==================
let selectedFile = null;
let currentStream = null;
let currentModel = null;
let realtimeInterval = null;

// ================== SWITCH INPUT TYPE ==================
function switchInputType(type, event) {
  document
    .querySelectorAll(".input-tab")
    .forEach((tab) => tab.classList.remove("active"));
  document
    .querySelectorAll(".input-content")
    .forEach((content) => content.classList.remove("active"));

  event.target.classList.add("active");
  document.getElementById(type + "-input").classList.add("active");

  resetInterface();
}

// ================== RESET INTERFACE ==================
function resetInterface() {
  selectedFile = null;
  resetResultPreview();
  enableImageButtons(false);
  enableVideoButtons(false);
  enableCameraButtons(false);
  if (realtimeInterval) {
    clearInterval(realtimeInterval);
    realtimeInterval = null;
  }
}

// ================== FILE INPUT HANDLERS ==================
document.getElementById("imageInput").addEventListener("change", function (e) {
  selectedFile = e.target.files[0];
  if (selectedFile) {
    displayOriginalPreview(selectedFile, "image");
    enableImageButtons(true);
  }
});

document.getElementById("videoInput").addEventListener("change", function (e) {
  selectedFile = e.target.files[0];
  if (selectedFile) {
    displayOriginalPreview(selectedFile, "video");
    enableVideoButtons(true);
  }
});

// ================== DISPLAY PREVIEW ==================
function displayOriginalPreview(file, type) {
  const reader = new FileReader();
  reader.onload = function (e) {
    const originalPreview = document.getElementById("originalPreview");
    if (type === "image") {
      originalPreview.innerHTML = `<img src="${e.target.result}" alt="Original image">`;
    } else if (type === "video") {
      originalPreview.innerHTML = `<video src="${e.target.result}" controls muted></video>`;
    }
    document.getElementById("previewSection").style.display = "grid";
    resetResultPreview();
  };
  reader.readAsDataURL(file);
}

function resetResultPreview() {
  document.getElementById("resultPreview").innerHTML =
    '<div class="preview-placeholder">Chờ xử lý...</div>';
}

// ================== ENABLE/DISABLE BUTTONS ==================
function enableImageButtons(enable) {
  document.getElementById("processYolov5Btn").disabled = !enable;
  document.getElementById("processYolov8Btn").disabled = !enable;
}

function enableVideoButtons(enable) {
  document.getElementById("videoYolov5Btn").disabled = !enable;
  document.getElementById("videoYolov8Btn").disabled = !enable;
}

function enableCameraButtons(enable) {
  document.getElementById("cameraYolov5Btn").disabled = !enable;
  document.getElementById("cameraYolov8Btn").disabled = !enable;
}

// ================== PROCESS IMAGE ==================
async function processWithModel(model) {
  if (!selectedFile) return;

  if (!selectedFile.type.startsWith("image/")) {
    alert("Chỉ hỗ trợ file ảnh!");
    hideLoading();
    return;
  }

  currentModel = model;
  showLoading(`Đang nhận diện với ${model.toUpperCase()}...`);

  try {
    const formData = new FormData();
    formData.append("file", selectedFile);

    const endpoint = model === "yolov5" ? "/predict_v5" : "/predict_v8";
    const response = await fetch(endpoint, {
      method: "POST",
      body: formData,
    });

    if (!response.ok) throw new Error(`Server error: ${model}`);

    const data = await response.json();
    displayResult(data.result_url, model);
  } catch (error) {
    console.error("Error processing model:", error);
    alert(`Lỗi xử lý ${model}: ` + error.message);
  } finally {
    hideLoading();
  }
}

// ================== DISPLAY RESULT ==================
function displayResult(resultUrl, model) {
  const resultPreview = document.getElementById("resultPreview");
  const resultTitle = document.getElementById("resultTitle");

  resultTitle.innerHTML = `🎯 Kết Quả ${model.toUpperCase()}`;
  resultPreview.innerHTML = `<img src="${resultUrl}?t=${new Date().getTime()}" alt="${model} result">`;
}

// ================== VIDEO PROCESS PLACEHOLDER ==================
function processVideo(model) {
  alert(`Chức năng xử lý video với ${model.toUpperCase()} đang phát triển.`);
}

// ================== CAMERA FUNCTIONS ==================
async function startCamera() {
  try {
    currentStream = await navigator.mediaDevices.getUserMedia({
      video: { width: 640, height: 480 },
    });
    const originalPreview = document.getElementById("originalPreview");
    const video = document.createElement("video");
    video.srcObject = currentStream;
    video.autoplay = true;
    video.muted = true;
    video.style.width = "100%";

    originalPreview.innerHTML = "";
    originalPreview.appendChild(video);

    document.getElementById("previewSection").style.display = "grid";
    document.getElementById("startCameraBtn").disabled = true;
    document.getElementById("stopCameraBtn").disabled = false;
    enableCameraButtons(true);
  } catch (error) {
    alert("Không thể truy cập camera: " + error.message);
  }
}

function stopCamera() {
  if (currentStream) {
    currentStream.getTracks().forEach((track) => track.stop());
    currentStream = null;
  }
  document.getElementById("originalPreview").innerHTML =
    '<div class="preview-placeholder">Chưa có nội dung</div>';
  document.getElementById("startCameraBtn").disabled = false;
  document.getElementById("stopCameraBtn").disabled = true;
  enableCameraButtons(false);
}

// ================== REALTIME CAMERA DETECTION PLACEHOLDER ==================
function startRealtimeDetection(model) {
  alert(
    `Chức năng nhận diện realtime với ${model.toUpperCase()} đang phát triển.`
  );
}

function stopRealtimeDetection() {
  if (realtimeInterval) {
    clearInterval(realtimeInterval);
    realtimeInterval = null;
  }
  alert("Dừng nhận diện realtime.");
}

// ================== LOADING UTILS ==================
function showLoading(message) {
  document.getElementById("loading").style.display = "block";
  document.getElementById("loadingText").innerText = message;
}

function hideLoading() {
  document.getElementById("loading").style.display = "none";
}
