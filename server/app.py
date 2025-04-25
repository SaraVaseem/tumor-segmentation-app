from flask import Flask, request, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
import pixellib
from pixellib.instance import custom_segmentation

app = Flask(__name__)
CORS(app)  # Allow cross-origin requests from React

UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "outputs"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Load PixelLib model
segment_image = custom_segmentation()
segment_image.inferConfig(num_classes=1, class_names = ["BG", "tumor"])
segment_image.load_model("(0.59)mask_rcnn_model.025-1.435254.h5")  # or your own model

@app.route("/upload", methods=["POST"])
def upload_file():
    file = request.files["file"]
    filename = secure_filename(file.filename)
    input_path = os.path.join(UPLOAD_FOLDER, filename)
    output_path = os.path.join(OUTPUT_FOLDER, f"segmented_{filename}")
    file.save(input_path)

    segment_image.segmentImage(input_path, output_image_name=output_path)

    return send_file(output_path, mimetype="image/png")

if __name__ == "__main__":
    app.run(debug=True)
