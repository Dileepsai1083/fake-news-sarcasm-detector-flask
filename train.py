import pandas as pd
import re
import pickle

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# =========================
# STEP 1: Load JSON dataset
# =========================
data = pd.read_json("Sarcasm_Headlines_Dataset.json", lines=True)

print("Columns:", data.columns)
print("Shape:", data.shape)

# Select columns
data = data[['headline', 'is_sarcastic']]
data.rename(columns={'headline': 'text', 'is_sarcastic': 'label'}, inplace=True)

# =========================
# STEP 2: SIMPLE STOPWORDS (NO NLTK)
# =========================
stop_words = set([
    'the','a','an','and','is','are','was','were','in','on','at',
    'to','for','of','with','that','this','it','as','by','from',
    'be','has','had','have','will','would','can','could','should'
])

def clean_text(text):
    text = str(text)
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z]", " ", text)
    text = text.lower()
    text = " ".join([w for w in text.split() if w not in stop_words])
    return text

data['clean_text'] = data['text'].apply(clean_text)

# =========================
# STEP 3: Features & labels
# =========================
X = data['clean_text']
y = data['label']

# =========================
# STEP 4: TF-IDF
# =========================
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(X)

# =========================
# STEP 5: Train-test split
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================
# STEP 6: Train model
# =========================
model = LogisticRegression()
model.fit(X_train, y_train)

# =========================
# STEP 7: Evaluate
# =========================
pred = model.predict(X_test)

print("\nAccuracy:", round(accuracy_score(y_test, pred), 2))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, pred))

# =========================
# STEP 8: Save model
# =========================
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("\n✅ Model saved successfully!")