import pandas as pd

df = pd.read_csv("train.csv")

df = df.rename(columns={"generated": "label", "text": "text"})
df["label"] = df["label"].map({"human": 0, "ai": 1})

df.to_csv("dataset.csv", index=False)