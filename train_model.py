import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle
import os
import pandas as pd

df = pd.read_csv("log_data.csv")
print(df.columns)

# Load data
df = pd.read_csv("log_data.csv")

X = df["log"]
y = df["label"]

# Vectorize
vectorizer = TfidfVectorizer()
X_vec = vectorizer.fit_transform(X)

# Train model
model = MultinomialNB()
model.fit(X_vec, y)

# Create model folder if not exist
os.makedirs("model", exist_ok=True)

# Save model
with open("model/log_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("model/vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("✅ Model trained and saved successfully!")
