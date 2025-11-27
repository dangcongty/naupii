from flask import Flask, request, jsonify
import time
from PIL import Image
import io

app = Flask(__name__)

@app.route("/api/process", methods=["POST"])
def process():
    if "file" not in request.files:
        return jsonify({"error": "no file"}), 400

    file = request.files["file"]
    img = Image.open(file.stream)

    time.sleep(3)

    return jsonify({
        "status": "ok",
        "width": img.width,
        "height": img.height
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=9000)
