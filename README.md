# AI Content Detector

A small Flask project that detects whether text, PDF, or Word documents are likely AI-generated and returns an AI score.

## Features
- Upload `.txt`, `.pdf`, or `.docx`
- Extract text automatically
- Predict AI-likeness with an ML classifier
- Show an AI score and a readable label
- Fallback heuristics when the model is not trained yet

## Project structure
- `app.py` — Flask app
- `train_model.py` — trains the model and creates `models/ai_detector.joblib`
- `detector.py` — loading, feature extraction, prediction
- `templates/index.html` — UI
- `static/style.css` — styling
- `static/script.js` — small client-side helpers

## Setup
```bash
pip install -r requirements.txt
python train_model.py
python app.py
```

Open `http://127.0.0.1:5000`

## Notes
- This is a baseline academic project, not a perfect detector.
- Accuracy depends heavily on the training data you use.
- For a stronger project, train on a large, diverse dataset of human and AI text.
