import json
import os
import subprocess
import tempfile
import time

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

# ---------------------------
# Routes
# ---------------------------
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
    
@app.route("/", methods=["GET", "POST"])
def wifi_page():
    ssid_list = scan_wifi_list()
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
camera = cv2.VideoCapture(0)   # dùng chung cho stream + capture
latest_frame = None            # lưu frame mới nhất từ stream


def gen_frames():
    global latest_frame, camera

    while True:
        success, frame = camera.read()
        if not success:
            break

        latest_frame = frame.copy()   # Lưu frame hiện tại

        ret, buffer = cv2.imencode('.jpg', frame)
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

    # ----  fake processing (3 seconds) ----
    time.sleep(3)

    # ---- save temp image ----
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
    cv2.imwrite(temp_file.name, latest_frame)

    return send_file(temp_file.name, mimetype='image/jpeg')

# ---------------------------
# Run
# ---------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=False)
