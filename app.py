from flask import Flask, render_template, request
import pickle
import re

app = Flask(__name__)

# Load model
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Same stopwords
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

@app.route("/")
def home():
    return render_template("index.html", prediction=None, text="")

@app.route("/predict", methods=["POST"])
def predict():
    user_text = request.form["text"]

    cleaned = clean_text(user_text)
    vect = vectorizer.transform([cleaned])
    pred = model.predict(vect)[0]

    result = "Sarcastic / Fake 😄" if pred == 1 else "Real / Normal 📰"

    return render_template("index.html",
                           prediction=result,
                           text=user_text)

if __name__ == "__main__":
    app.run(debug=True)