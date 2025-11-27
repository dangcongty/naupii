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

# ========== MOCK MODE (để test trên server) ==========
MOCK_MODE = os.environ.get("MOCK_MODE", "false").lower() == "true"

app = Flask(__name__)

# ---------------------------
# Get current connected SSID
# ---------------------------
def get_current_ssid():
    try:
        out = subprocess.check_output(
            ["nmcli", "-t", "-f", "ACTIVE,SSID", "dev", "wifi"],
            text=True
        )
        for line in out.splitlines():
            parts = line.split(":")
            if parts[0] == "yes":
                return parts[1]
    except:
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
    try:
        cmd = ["nmcli", "dev", "wifi", "connect", ssid, "password", password]
        result = subprocess.run(cmd, capture_output=True, text=True)
        return result.returncode == 0
    except:
        return False

def scan_wifi_list():
    """Return list of available SSIDs."""
    try:
        out = subprocess.check_output(
            ["nmcli", "-t", "-f", "SSID", "dev", "wifi"], text=True
        )
        ssids = []
        for line in out.splitlines():
            if line.strip() != "":
                ssids.append(line.strip())
        return sorted(list(set(ssids)))
    except:
        return []

# ---------------------------
# Routes
# ---------------------------
#  

@app.route("/", methods=["GET", "POST"])
def wifi_page():
    ssid_list = scan_wifi_list()
    print(f"DEBUG: Found {len(ssid_list)} networks: {ssid_list}") 
    saved_ssid, saved_password = load_wifi_config()
    current_ssid = get_current_ssid()

    # ----------------------------------------
    # Nếu đã kết nối đúng WiFi trước đó → sang stream luôn
    # ----------------------------------------
    if current_ssid and saved_ssid and current_ssid == saved_ssid:
        return redirect("/stream")

    # ----------------------------------------
    # Submit form connect WiFi
    # ----------------------------------------
    if request.method == "POST":
        ssid = request.form.get("ssid")
        password = request.form.get("password")

        ok = connect_wifi(ssid, password)
        if ok:
            save_wifi_config(ssid, password)      # <-- Lưu lại
            return redirect("/stream")
        else:
            return render_template("wifi.html",
                                   ssid_list=ssid_list,
                                   saved_ssid=saved_ssid,
                                   error=f"Failed to connect to {ssid}")

    # ----------------------------------------
    # GET request: hiển thị trang chọn WiFi
    # ----------------------------------------
    return render_template("wifi.html",
                           ssid_list=ssid_list,
                           saved_ssid=saved_ssid)


@app.route("/stream")
def stream_page():
    return render_template("stream.html")


# ---------------------------
# MJPEG video generator
# ---------------------------
if MOCK_MODE:
    print("[MOCK] Using fake camera (no /dev/video0)")
    camera = None
else:
    camera = cv2.VideoCapture(0)

latest_frame = None


def gen_frames():
    global latest_frame, camera

    while True:
        if MOCK_MODE:
            # Tạo fake frame (gradient màu)
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            frame[:, :, 0] = np.linspace(0, 255, 640, dtype=np.uint8)  # Blue gradient
            frame[:, :, 1] = 128  # Green
            frame[:, :, 2] = np.linspace(255, 0, 640, dtype=np.uint8)  # Red gradient
            
            # Thêm text timestamp
            timestamp = time.strftime("%H:%M:%S")
            cv2.putText(frame, f"MOCK CAMERA - {timestamp}", (50, 240),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
            
            latest_frame = frame.copy()
            time.sleep(0.03)  # ~30 FPS
        else:
            success, frame = camera.read()
            if not success:
                break
            latest_frame = frame.copy()

        ret, buffer = cv2.imencode('.jpg', latest_frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


@app.route("/video_feed")
def video_feed():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# -----------------------------------
# PROCESS CAPTURE API
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
    app.run(host="0.0.0.0", port=7000, debug=False)
