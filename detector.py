import os
import re
import joblib
from docx import Document
from pypdf import PdfReader

MODEL_PATH = os.path.join("models", "model.joblib")
VECTORIZER_PATH = os.path.join("models", "vectorizer.joblib")

class AIDetector:
    def __init__(self, model_path: str = MODEL_PATH, vectorizer_path: str = VECTORIZER_PATH):
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model not found at {model_path}. Run train_model.py first."
            )
        if not os.path.exists(vectorizer_path):
            raise FileNotFoundError(
                f"Vectorizer not found at {vectorizer_path}. Run train_model.py first."
            )
        self.model = joblib.load(model_path)
        self.vectorizer = joblib.load(vectorizer_path)

    def extract_text_from_txt(self, file_path: str) -> str:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()

    def extract_text_from_pdf(self, file_path: str) -> str:
        text = []
        reader = PdfReader(file_path)
        for page in reader.pages:
            page_text = page.extract_text() or ""
            text.append(page_text)
        return "\n".join(text)

    def extract_text_from_docx(self, file_path: str) -> str:
        doc = Document(file_path)
        return "\n".join([p.text for p in doc.paragraphs])

    def split_sentences(self, text: str):
        text = re.sub(r"\s+", " ", text).strip()
        if not text:
            return []
        parts = re.split(r"(?<=[.!?])\s+", text)
        sentences = [s.strip() for s in parts if len(s.strip()) > 10]
        return sentences if sentences else [text]

    def clean_text(self, text: str) -> str:
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def text_stats(self, text: str):
        sentences = self.split_sentences(text)
        words = re.findall(r"\b\w+\b", text.lower())

        word_count = len(words)
        sentence_count = len(sentences)
        avg_sentence_length = round(word_count / sentence_count, 2) if sentence_count else 0

        unique_words = len(set(words))
        lexical_diversity = round(unique_words / word_count, 3) if word_count else 0

        word_freq = {}
        for w in words:
            word_freq[w] = word_freq.get(w, 0) + 1
        repeated_words = sum(1 for _, c in word_freq.items() if c > 2)
        repetition_ratio = round(repeated_words / len(word_freq), 3) if word_freq else 0

        return {
            "word_count": word_count,
            "sentence_count": sentence_count,
            "avg_sentence_length": avg_sentence_length,
            "lexical_diversity": lexical_diversity,
            "repetition_ratio": repetition_ratio
        }

    def explanation_signals(self, stats):
        signals = []

        if stats["lexical_diversity"] and stats["lexical_diversity"] < 0.55:
            signals.append("Low lexical diversity")
        if stats["repetition_ratio"] > 0.12:
            signals.append("Repeated word usage detected")
        if stats["avg_sentence_length"] and 12 <= stats["avg_sentence_length"] <= 25:
            signals.append("Very uniform sentence style")
        if stats["sentence_count"] < 3:
            signals.append("Very short text; confidence may be lower")

        return signals

    def score_label(self, score: float):
        if score >= 0.70:
            return "Likely AI"
        if score >= 0.40:
            return "Mixed"
        return "Likely Human"

    def predict_ai_probability(self, text: str) -> float:
        X = self.vectorizer.transform([text])
        prediction = int(self.model.predict(X)[0])
        if hasattr(self.model, "predict_proba"):
            return float(self.model.predict_proba(X)[0][1])
        return float(prediction)

    def analyze_text(self, text: str):
        text = self.clean_text(text)
        if not text:
            return {
                "overall_score": 0.0,
                "label": "No Text",
                "sentence_results": [],
                "stats": {},
                "signals": []
            }

        overall_prob = self.predict_ai_probability(text)
        sentences = self.split_sentences(text)

        sentence_results = []
        for sentence in sentences:
            prob = self.predict_ai_probability(sentence)
            sentence_results.append({
                "sentence": sentence,
                "ai_score": round(prob, 4),
                "label": self.score_label(prob)
            })

        stats = self.text_stats(text)
        signals = self.explanation_signals(stats)

        return {
            "overall_score": round(overall_prob, 4),
            "label": self.score_label(overall_prob),
            "sentence_results": sentence_results,
            "stats": stats,
            "signals": signals
        }

    def analyze_file(self, file_path: str, file_ext: str):
        file_ext = file_ext.lower()

        if file_ext == ".txt":
            text = self.extract_text_from_txt(file_path)
        elif file_ext == ".pdf":
            text = self.extract_text_from_pdf(file_path)
        elif file_ext == ".docx":
            text = self.extract_text_from_docx(file_path)
        else:
            raise ValueError("Unsupported file type")

        return self.analyze_text(text)
