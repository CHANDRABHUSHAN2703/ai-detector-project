import csv
import random

human_samples = [
    "I went to the market yesterday and bought some fresh vegetables.",
    "My college experience has been full of learning and growth.",
    "Sometimes I prefer reading books over watching movies.",
    "I enjoy spending time with my friends during weekends.",
    "The weather today is quite pleasant and refreshing."
]

ai_samples = [
    "Artificial intelligence systems are increasingly being utilized to optimize complex workflows and enhance productivity.",
    "The rapid evolution of machine learning models has enabled significant advancements in natural language processing.",
    "AI-generated text often exhibits high coherence and structured sentence formation.",
    "Advanced algorithms can generate contextually relevant and semantically meaningful content.",
    "Modern AI technologies are capable of producing human-like responses across various domains."
]

def generate_dataset(file_name="dataset.csv", size=1000):
    data = []

    for _ in range(size // 2):
        data.append([random.choice(human_samples), 0])
        data.append([random.choice(ai_samples), 1])

    random.shuffle(data)

    with open(file_name, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["text", "label"])
        writer.writerows(data)

    print(f"Dataset generated with {size} samples")

if __name__ == "__main__":
    generate_dataset()