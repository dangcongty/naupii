import json
import os
import subprocess
import tempfile
import time
import numpy as np
import requests
import cv2
from flask import (
    Flask,
    Response,
    jsonify,
    redirect,
    render_template,
    request,
    send_file,
)

CONFIG_FILE = "wifi_config.json"
BACKEND_URL = "http://118.69.83.17:9000/api/process" 

# ========== CAMERA CONFIG ==========
# Thay đổi resolution ở đây để có góc nhìn rộng hơn
CAMERA_WIDTH = 3264   # Thử: 3264, 1640, 1920
CAMERA_HEIGHT = 2464  # Thử: 2464, 1232, 1080
CAMERA_FPS = 21       # 21 cho 3264x2464, 30 cho các resolution khác

# ========== MOCK MODE ==========
MOCK_MODE = os.environ.get("MOCK_MODE", "false").lower() == "true"

app = Flask(__name__)

# ---------------------------
# Get current connected SSID
# ---------------------------
def get_current_ssid():
    if MOCK_MODE:
        return "MockWiFi_Home"
    try:
        result = subprocess.run(
            ["nmcli", "-t", "-f", "ACTIVE,SSID", "dev", "wifi"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        for line in result.stdout.splitlines():
            parts = line.split(":")
            if len(parts) >= 2 and parts[0] == "yes":
                return parts[1]
    except Exception as e:
        print(f"[ERROR] get_current_ssid failed: {e}")
        return None
    return None


def save_wifi_config(ssid, password):
    data = {
        "saved_ssid": ssid,
        "saved_password": password
    }
    with open(CONFIG_FILE, "w") as f:
        json.dump(data, f)


def load_wifi_config():
    if not os.path.exists(CONFIG_FILE):
        return None, None
    try:
        with open(CONFIG_FILE, "r") as f:
            data = json.load(f)
            return data.get("saved_ssid"), data.get("saved_password")
    except:
        return None, None

# ---------------------------
# Connect to WiFi
# ---------------------------
def connect_wifi(ssid, password):
    if MOCK_MODE:
        print(f"[MOCK] Connecting to {ssid}...")
        time.sleep(1)
        return True
    try:
        cmd = ["sudo", "nmcli", "dev", "wifi", "connect", ssid, "password", password]
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        if result.returncode == 0:
            print(f"[INFO] Connected to {ssid}")
            return True
        else:
            print(f"[ERROR] Connect failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"[ERROR] connect_wifi exception: {e}")
        return False

def scan_wifi_list():
    """Return list of available SSIDs."""
    if MOCK_MODE:
        return ["MockWiFi_Home", "MockWiFi_Office", "MockWiFi_Guest", "TestNetwork_5G"]
    
    try:
        # Rescan WiFi (with sudo, Python 3.6 compatible)
        subprocess.run(
            ["sudo", "nmcli", "device", "wifi", "rescan"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=5
        )
        time.sleep(2)
        
        # List WiFi networks
        result = subprocess.run(
            ["sudo", "nmcli", "-t", "-f", "SSID", "device", "wifi", "list"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=10,
            universal_newlines=True
        )
        
        if result.returncode != 0:
            print(f"[ERROR] nmcli failed: {result.stderr}")
            return []
        
        ssids = []
        for line in result.stdout.splitlines():
            line = line.strip()
            if line and line != "" and line != "--":
                ssids.append(line)
        
        unique_ssids = sorted(list(set(ssids)))
        print(f"[DEBUG] Found {len(unique_ssids)} WiFi networks: {unique_ssids}")
        return unique_ssids
        
    except subprocess.TimeoutExpired:
        print("[ERROR] WiFi scan timeout")
        return []
    except Exception as e:
        print(f"[ERROR] WiFi scan failed: {e}")
        return []

# ---------------------------
# Routes
# ---------------------------

@app.route("/", methods=["GET", "POST"])
def wifi_page():
    ssid_list = scan_wifi_list()
    saved_ssid, saved_password = load_wifi_config()
    current_ssid = get_current_ssid()
    
    print(f"[DEBUG] WiFi scan result: {len(ssid_list)} networks")
    print(f"[DEBUG] Current SSID: {current_ssid}")
    print(f"[DEBUG] Saved SSID: {saved_ssid}")

    # Nếu đã kết nối đúng WiFi trước đó → sang stream luôn
    if current_ssid and saved_ssid and current_ssid == saved_ssid:
        return redirect("/stream")

    # Submit form connect WiFi
    if request.method == "POST":
        ssid = request.form.get("ssid")
        password = request.form.get("password")

        ok = connect_wifi(ssid, password)
        if ok:
            save_wifi_config(ssid, password)
            return redirect("/stream")
        else:
            return render_template("wifi.html",
                                   ssid_list=ssid_list,
                                   saved_ssid=saved_ssid,
                                   error=f"Failed to connect to {ssid}")

    # GET request: hiển thị trang chọn WiFi
    return render_template("wifi.html",
                           ssid_list=ssid_list,
                           saved_ssid=saved_ssid)


@app.route("/stream")
def stream_page():
    return render_template("stream.html")


# ---------------------------
# MJPEG video generator
# ---------------------------
def get_jetson_camera():
    """Open camera with GStreamer pipeline for Jetson"""
    
    # Try với resolution đã cấu hình
    gst_str = (
        f"nvarguscamerasrc ! "
        f"video/x-raw(memory:NVMM), width=(int){CAMERA_WIDTH}, height=(int){CAMERA_HEIGHT}, "
        f"format=(string)NV12, framerate=(fraction){CAMERA_FPS}/1 ! "
        f"nvvidconv ! video/x-raw, format=(string)BGRx ! "
        f"videoconvert ! video/x-raw, format=(string)BGR ! appsink"
    )
    
    print(f"[INFO] Trying CSI camera: {CAMERA_WIDTH}x{CAMERA_HEIGHT}@{CAMERA_FPS}fps")
    cap = cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)
    if cap.isOpened():
        print(f"[INFO] ✓ CSI camera opened: {CAMERA_WIDTH}x{CAMERA_HEIGHT}@{CAMERA_FPS}fps")
        return cap
    
    # Fallback 1: Try 1640x1232 (wide FOV)
    print("[WARNING] Trying fallback: 1640x1232@30fps")
    gst_str = (
        "nvarguscamerasrc ! "
        "video/x-raw(memory:NVMM), width=(int)1640, height=(int)1232, "
        "format=(string)NV12, framerate=(fraction)30/1 ! "
        "nvvidconv ! video/x-raw, format=(string)BGRx ! "
        "videoconvert ! video/x-raw, format=(string)BGR ! appsink"
    )
    cap = cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)
    if cap.isOpened():
        print("[INFO] ✓ CSI camera opened with 1640x1232@30fps")
        return cap
    
    # Fallback 2: USB camera
    print("[WARNING] CSI camera failed, trying USB camera")
    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        print("[INFO] ✓ USB camera opened")
        return cap
    
    print("[ERROR] ✗ No camera found!")
    return None

if MOCK_MODE:
    print("[MOCK] Using fake camera (no /dev/video0)")
    camera = None
else:
    camera = get_jetson_camera()

latest_frame = None


def gen_frames():
    global latest_frame, camera

    while True:
        if MOCK_MODE:
            # Tạo fake frame (gradient màu)
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            frame[:, :, 0] = np.linspace(0, 255, 640, dtype=np.uint8)
            frame[:, :, 1] = 128
            frame[:, :, 2] = np.linspace(255, 0, 640, dtype=np.uint8)
            
            timestamp = time.strftime("%H:%M:%S")
            cv2.putText(frame, f"MOCK CAMERA - {timestamp}", (50, 240),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
            
            latest_frame = frame.copy()
            time.sleep(0.03)
        else:
            if camera is None:
                break
            success, frame = camera.read()
            if not success:
                break
            latest_frame = frame.copy()

        ret, buffer = cv2.imencode('.jpg', latest_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


@app.route("/video_feed")
def video_feed():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# -----------------------------------
# CAPTURE API
# -----------------------------------

@app.route("/capture", methods=["POST"])
def capture():
    global latest_frame

    if latest_frame is None:
        return jsonify({"error": "No frame available"}), 500

    try:
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
        cv2.imwrite(temp_file.name, latest_frame)
        
        with open(temp_file.name, 'rb') as f:
            files = {'file': ('capture.jpg', f, 'image/jpeg')}
            response = requests.post(BACKEND_URL, files=files, timeout=30)
        
        os.unlink(temp_file.name)
        
        if response.status_code == 200:
            result = response.json()
            return jsonify({
                "success": True,
                "result": result
            })
        else:
            return jsonify({"error": "Backend processing failed"}), 500
            
    except requests.exceptions.RequestException as e:
        return jsonify({"error": f"Cannot connect to backend: {str(e)}"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ---------------------------
# Run
# ---------------------------
if __name__ == "__main__":
    if MOCK_MODE:
        print("\n" + "="*50)
        print("🧪 RUNNING IN MOCK MODE (for testing on server)")
        print("="*50 + "\n")
    
    print(f"\n📷 Camera Config: {CAMERA_WIDTH}x{CAMERA_HEIGHT}@{CAMERA_FPS}fps\n")
    app.run(host="0.0.0.0", port=8000, debug=False)