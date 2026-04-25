# AI Content Detector

A Flask-based web application that detects whether text, PDF, or Word files are AI-generated or human-written using Machine Learning.

---

## Features

*  Upload TXT, PDF, DOCX files
*  AI probability score
*  Machine learning-based classification
*  Simple web interface

---

##  How It Works

* Extracts text from files
* Applies NLP preprocessing
* Uses ML model (TF-IDF + Logistic Regression)
* Outputs AI probability score

---

## 📸 Demo Screenshots



---

##  Tech Stack

* Python
* Flask
* scikit-learn
* HTML/CSS

---

## ▶️ How to Run

```bash
pip install -r requirements.txt
python train_model.py
python app.py
```

---

## Sample Output

| Input Type | Result              |
| ---------- | ------------------- |
| Human Text | 0.25 (Likely Human) |
| AI Text    | 0.89 (Likely AI)    |

---

##  Limitations

* Not 100% accurate
* Can be fooled by paraphrasing
* Works best on longer text

---

## Future Improvements

* BERT-based detection
* Sentence-level highlighting
* Cloud deployment

---

## Author

Chandra Bhushan
