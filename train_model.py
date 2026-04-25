from pathlib import Path
import random
import re

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


MODEL_PATH = Path("models/ai_detector.joblib")


HUMAN_SAMPLES = [
    "I went to the market yesterday and bought tomatoes, onions, and a small packet of tea.",
    "The bus was late, so I missed the first class and had to explain it to the teacher.",
    "My grandmother told me stories about the village when she was young, and I still remember them.",
    "After dinner, we sat outside and talked for nearly an hour about exams and family.",
    "I wrote the assignment in the afternoon, but I had to correct a few mistakes before submitting it.",
    "The room was noisy, and it was hard to focus until everyone left for lunch.",
    "Last Sunday, we visited our relatives, drank chai, and watched the children play cricket.",
    "The road near the bridge is broken, so the driver took a longer route through the town.",
    "I felt tired after walking in the heat, but the fresh coconut water helped a lot.",
    "When the power went out, we lit candles and finished our conversation in the dark.",
    "The teacher asked a simple question, but the class became quiet for a moment.",
    "I checked the clock twice because I was worried about being late for the meeting.",
    "Her handwriting is neat, though she still makes a few spelling mistakes in English.",
    "I tried the new recipe at home, but it turned out too spicy for my family.",
    "The exam was difficult, especially the section on probability and graphs.",
]

AI_SAMPLES = [
    "In today's rapidly evolving digital landscape, it is essential to leverage innovative strategies to maximize efficiency and foster sustainable growth.",
    "This article explores the transformative potential of advanced technologies in modern workflows and highlights key considerations for implementation.",
    "From a strategic perspective, the proposed framework enables seamless integration, improved scalability, and enhanced user satisfaction.",
    "It is important to note that the solution provides a comprehensive and robust approach to solving the identified problem.",
    "The analysis demonstrates that careful optimization can significantly improve outcomes across multiple domains and use cases.",
    "Overall, the system presents a balanced combination of flexibility, reliability, and performance for diverse operational requirements.",
    "By adopting a structured methodology, organizations can achieve consistent results while minimizing operational inefficiencies.",
    "The following discussion presents a clear overview of the process, key benefits, and practical implications of the proposed model.",
    "Furthermore, the evidence suggests that a data-driven approach can support informed decision-making and long-term success.",
    "In conclusion, the findings indicate strong potential for adoption, provided that implementation is supported by continuous evaluation.",
    "This response is organized into sections to improve readability, clarity, and logical progression of ideas.",
    "The objective is to deliver a concise yet comprehensive explanation that addresses the user's request effectively.",
    "Considering the available information, the most appropriate next step is to evaluate the options and select the optimal solution.",
    "The model generates coherent, polished, and grammatically consistent output with a high degree of predictability.",
    "As a result, the approach is suitable for environments that require clarity, consistency, and structured communication.",
]


def augment_text(text: str, label: int) -> str:
    extras = [
        " therefore",
        " moreover",
        " in addition",
        " however",
        " because of this",
        " on the other hand",
        " for example",
        " in summary",
    ]
    if label == 1:
        # AI-like augmentations: extra polish and repetition
        additions = random.sample(extras, k=2)
        return text + ". " + " ".join(additions).capitalize() + "."
    # Human-like augmentations: messy, natural variation
    endings = [
        " I think.",
        " maybe later.",
        " not sure why.",
        " that's all.",
        " it was okay.",
        " honestly.",
    ]
    return text + random.choice(endings)


def build_dataset():
    texts = []
    labels = []

    for s in HUMAN_SAMPLES:
        for _ in range(8):
            texts.append(augment_text(s, 0))
            labels.append(0)

    for s in AI_SAMPLES:
        for _ in range(10):
            texts.append(augment_text(s, 1))
            labels.append(1)

    random.seed(42)
    combined = list(zip(texts, labels))
    random.shuffle(combined)
    texts, labels = zip(*combined)
    return list(texts), list(labels)


def main():
    texts, labels = build_dataset()

    pipeline = Pipeline([
        ("vectorizer", TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            min_df=1,
            max_features=10000,
            stop_words="english"
        )),
        ("classifier", LogisticRegression(max_iter=2000))
    ])

    pipeline.fit(texts, labels)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    bundle = {
        "vectorizer": pipeline.named_steps["vectorizer"],
        "classifier": pipeline.named_steps["classifier"],
        "class_map": [0, 1],  # 0 = human, 1 = ai
    }
    joblib.dump(bundle, MODEL_PATH)

    print(f"Saved model to {MODEL_PATH}")
    print("Training samples:", len(texts))


if __name__ == "__main__":
    main()
