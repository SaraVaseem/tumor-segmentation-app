from flask import Flask, request, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
import pixellib
from pixellib.instance import instance_segmentation

app = Flask(__name__)
CORS(app)  # Allow cross-origin requests from React

UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "outputs"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Load PixelLib model
segment_image = instance_segmentation()
segment_image.load_model("mask_rcnn_coco.h5")  # or your own model

@app.route("/upload", methods=["POST"])
def upload_file():
    file = request.files["file"]
    filename = secure_filename(file.filename)
    input_path = os.path.join(UPLOAD_FOLDER, filename)
    output_path = os.path.join(OUTPUT_FOLDER, f"segmented_{filename}")
    file.save(input_path)

    segment_image.segmentImage(image_path=input_path, output_image_name=output_path)

    return send_file(output_path, mimetype="image/png")

if __name__ == "__main__":
    app.run(debug=True)
