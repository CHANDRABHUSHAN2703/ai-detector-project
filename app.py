import os

# if not os.path.exists("models/ai_detector.joblib"):
    # import train_model
import uuid
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from detector import AIDetector

app = Flask(__name__)
detector = AIDetector()
app.config["UPLOAD_FOLDER"] = "uploads"
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

ALLOWED_EXTENSIONS = {".txt", ".pdf", ".docx"}

os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

# if not os.path.exists("models/ai_detector.joblib"):
#     import train_model

from detector import AIDetector
from detector import AIDetector

detector = AIDetector()
def allowed_file(filename):
    return "." in filename and os.path.splitext(filename)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/analyze", methods=["POST"])
def analyze():
    try:
        text_input = request.form.get("text", "").strip()
        uploaded_file = request.files.get("file")

        if text_input:
            result = detector.analyze_text(text_input)
            return jsonify({"success": True, "result": result})

        if uploaded_file and uploaded_file.filename:
            if not allowed_file(uploaded_file.filename):
                return jsonify({"success": False, "error": "Only .txt, .pdf, and .docx files are allowed"})

            ext = os.path.splitext(uploaded_file.filename)[1].lower()
            safe_name = secure_filename(uploaded_file.filename)
            unique_name = f"{uuid.uuid4().hex}_{safe_name}"
            file_path = os.path.join(app.config["UPLOAD_FOLDER"], unique_name)
            uploaded_file.save(file_path)

            result = detector.analyze_file(file_path, ext)

            try:
                os.remove(file_path)
            except Exception:
                pass

            return jsonify({"success": True, "result": result})

        return jsonify({"success": False, "error": "Please enter text or upload a file"})

    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
