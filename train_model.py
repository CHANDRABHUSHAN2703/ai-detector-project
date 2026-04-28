import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

DATASET_PATH = "dataset.csv"
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "ai_detector.joblib")
VECTORIZER_PATH = os.path.join(MODEL_DIR, "vectorizer.joblib")
CLASSIFIER_PATH = os.path.join(MODEL_DIR, "model.joblib")

def main():
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"{DATASET_PATH} not found")

    data = pd.read_csv(DATASET_PATH)

    if "text" not in data.columns or "label" not in data.columns:
        raise ValueError("dataset.csv must contain 'text' and 'label' columns")

    data = data.dropna(subset=["text", "label"])
    data["text"] = data["text"].astype(str).str.strip()
    data = data[data["text"] != ""]

    X = data["text"]
    y = data["label"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    vectorizer = TfidfVectorizer(
        max_features=8000,
        ngram_range=(1, 2),
        stop_words="english"
    )
    model = LogisticRegression(
        max_iter=3000,
        class_weight="balanced"
    )

    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    model.fit(X_train_vec, y_train)

    y_pred = model.predict(X_test_vec)
    acc = accuracy_score(y_test, y_pred)

    print("\n=== Model Evaluation ===")
    print("Accuracy:", round(acc, 4))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(vectorizer, VECTORIZER_PATH)
    joblib.dump(model, CLASSIFIER_PATH)

    # Backward-compatible pipeline artifact used by existing app startup checks.
    legacy_pipeline = Pipeline([
        ("tfidf", vectorizer),
        ("clf", model),
    ])
    joblib.dump(legacy_pipeline, MODEL_PATH)

    print(f"Vectorizer saved to: {VECTORIZER_PATH}")
    print(f"Model saved to: {CLASSIFIER_PATH}")
    print(f"\nModel saved to: {MODEL_PATH}")

if __name__ == "__main__":
    main()
