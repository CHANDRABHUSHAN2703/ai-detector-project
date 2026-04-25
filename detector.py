import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
from pypdf import PdfReader
from docx import Document


MODEL_PATH = Path("models/ai_detector.joblib")


@dataclass
class PredictionResult:
    label: str
    ai_score: float
    classifier_prob_ai: float
    heuristic_score: float
    confidence: float
    text_length: int
    extracted_from: str
    details: Dict[str, float]


def clean_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_text_from_txt(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        return clean_text(f.read())


def extract_text_from_docx(file_path: str) -> str:
    doc = Document(file_path)
    parts = [p.text for p in doc.paragraphs if p.text.strip()]
    return clean_text("\n".join(parts))


def extract_text_from_pdf(file_path: str) -> str:
    reader = PdfReader(file_path)
    pages = []
    for page in reader.pages:
        pages.append(page.extract_text() or "")
    return clean_text("\n".join(pages))


def extract_text(file_path: str) -> Tuple[str, str]:
    suffix = Path(file_path).suffix.lower()
    if suffix == ".txt":
        return extract_text_from_txt(file_path), "txt"
    if suffix == ".docx":
        return extract_text_from_docx(file_path), "docx"
    if suffix == ".pdf":
        return extract_text_from_pdf(file_path), "pdf"
    raise ValueError("Unsupported file type. Please upload a .txt, .pdf, or .docx file.")


def sentence_split(text: str) -> List[str]:
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in sentences if s.strip()]


def word_tokens(text: str) -> List[str]:
    return re.findall(r"[A-Za-z']+", text.lower())


def heuristic_features(text: str) -> Dict[str, float]:
    words = word_tokens(text)
    sentences = sentence_split(text)
    total_words = len(words)
    total_sentences = max(len(sentences), 1)
    unique_words = len(set(words))

    avg_sentence_length = total_words / total_sentences
    ttr = unique_words / max(total_words, 1)  # type-token ratio
    repeat_ratio = 0.0
    if total_words > 10:
        repeat_ratio = 1.0 - (unique_words / total_words)

    punctuation_count = len(re.findall(r"[.,;:!?]", text))
    punctuation_density = punctuation_count / max(len(text), 1)

    long_word_ratio = 0.0
    if total_words:
        long_word_ratio = sum(1 for w in words if len(w) >= 8) / total_words

    avg_sentence_var = 0.0
    if len(sentences) >= 2:
        lengths = [len(word_tokens(s)) for s in sentences]
        avg = sum(lengths) / len(lengths)
        avg_sentence_var = sum((x - avg) ** 2 for x in lengths) / len(lengths)

    ai_like = 0.0
    ai_like += max(0.0, min(1.0, (avg_sentence_length - 10) / 25)) * 0.30
    ai_like += max(0.0, min(1.0, (0.45 - ttr) / 0.25)) * 0.25
    ai_like += max(0.0, min(1.0, repeat_ratio / 0.35)) * 0.20
    ai_like += max(0.0, min(1.0, (0.03 - punctuation_density) / 0.02)) * 0.10
    ai_like += max(0.0, min(1.0, (long_word_ratio - 0.18) / 0.20)) * 0.15

    ai_like = float(max(0.0, min(1.0, ai_like)))

    return {
        "avg_sentence_length": float(avg_sentence_length),
        "type_token_ratio": float(ttr),
        "repeat_ratio": float(repeat_ratio),
        "punctuation_density": float(punctuation_density),
        "long_word_ratio": float(long_word_ratio),
        "sentence_length_variance": float(avg_sentence_var),
        "heuristic_score": ai_like,
    }


def load_model():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)


def predict_text(text: str) -> PredictionResult:
    text = clean_text(text)
    if not text:
        raise ValueError("No text found after extraction.")

    model_bundle = load_model()
    feats = heuristic_features(text)

    classifier_prob_ai = 0.5
    confidence = 0.0

    if model_bundle is not None:
        vectorizer = model_bundle["vectorizer"]
        classifier = model_bundle["classifier"]
        probs = classifier.predict_proba(vectorizer.transform([text]))[0]
        class_map = model_bundle["class_map"]
        ai_index = class_map.index(1)
        classifier_prob_ai = float(probs[ai_index])
        confidence = float(abs(classifier_prob_ai - 0.5) * 2.0)

    ai_score = 0.7 * classifier_prob_ai + 0.3 * feats["heuristic_score"]
    ai_score = float(max(0.0, min(1.0, ai_score)))

    if ai_score >= 0.70:
        label = "Likely AI-generated"
    elif ai_score >= 0.45:
        label = "Mixed / Uncertain"
    else:
        label = "Likely human-written"

    return PredictionResult(
        label=label,
        ai_score=round(ai_score * 100, 2),
        classifier_prob_ai=round(classifier_prob_ai * 100, 2),
        heuristic_score=round(feats["heuristic_score"] * 100, 2),
        confidence=round(confidence * 100, 2),
        text_length=len(text),
        extracted_from="text",
        details={k: round(v, 4) if isinstance(v, float) else v for k, v in feats.items()},
    )
