from pathlib import Path
import os
import uuid

from flask import Flask, flash, jsonify, render_template, request

from detector import extract_text, predict_text


UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

app = Flask(__name__)
app.secret_key = "ai-detector-secret-key"
app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024  # 20 MB


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    try:
        file = request.files.get("file")
        manual_text = request.form.get("text", "").strip()

        extracted = ""
        source = "text"

        if file and file.filename:
            ext = Path(file.filename).suffix.lower()
            if ext not in {".txt", ".pdf", ".docx"}:
                return jsonify({"error": "Only .txt, .pdf, and .docx files are supported."}), 400

            safe_name = f"{uuid.uuid4().hex}{ext}"
            save_path = UPLOAD_DIR / safe_name
            file.save(save_path)

            extracted, source = extract_text(str(save_path))
            try:
                os.remove(save_path)
            except OSError:
                pass
        elif manual_text:
            extracted = manual_text
            source = "text"
        else:
            return jsonify({"error": "Please enter text or upload a file."}), 400

        result = predict_text(extracted)
        result.extracted_from = source

        return jsonify({
            "label": result.label,
            "ai_score": result.ai_score,
            "classifier_prob_ai": result.classifier_prob_ai,
            "heuristic_score": result.heuristic_score,
            "confidence": result.confidence,
            "text_length": result.text_length,
            "extracted_from": source,
            "details": result.details,
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(debug=True)
