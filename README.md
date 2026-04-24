# 📰 Fake News / Sarcasm Detector (ML + Flask)

This project is a Machine Learning-based web application that detects whether a news headline is **Fake (Sarcastic 😄)** or **Real (Normal 📰)**.

---

## 🚀 Features

* 🔮 Real-time prediction using Flask
* 🧠 NLP-based text classification
* 🧹 Text preprocessing (cleaning + stopwords removal)
* ⚡ TF-IDF vectorization
* 🎯 Logistic Regression model
* 🎨 Modern UI with Bootstrap + animations
* 🔄 Clear button to reset input

---

## 🛠 Tech Stack

* Python
* Pandas, NumPy
* Scikit-learn
* Flask
* HTML, CSS, Bootstrap

---

## 📊 Dataset

Dataset used: **Sarcasm Headlines Dataset**

* Source: Kaggle
* Original data from:

  * The Onion (sarcastic news)
  * HuffPost (real news)

### Labels

* `1` → Sarcastic (Fake)
* `0` → Not Sarcastic (Real)

---

## ⚙️ How It Works

1. Load JSON dataset
2. Extract headlines and labels
3. Clean text (remove symbols, lowercase, stopwords)
4. Convert text into numerical vectors using TF-IDF
5. Train Logistic Regression model
6. Save model using pickle
7. Build Flask UI for prediction

---

## 📁 Project Structure

```bash
fake-news-sarcasm-detector-flask/
│── app.py
│── train.py
│── model.pkl
│── vectorizer.pkl
│── Sarcasm_Headlines_Dataset.json
│── requirements.txt
│── README.md
│
└── templates/
    └── index.html
```

---

## ▶️ Run Locally

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Train model

```bash
python train.py
```

### 3. Run Flask app

```bash
python app.py
```

### 4. Open browser

```
http://127.0.0.1:5000/
```

---

## 🧪 Example

| Input                          | Output  |
| ------------------------------ | ------- |
| BREAKING: Aliens landed in USA | Fake 😄 |
| Government passes new law      | Real 📰 |

---

## 📸 Screenshots
real:

<img width="2134" height="1184" alt="real news" src="https://github.com/user-attachments/assets/4fbc864e-d78f-4ef1-8f17-a70e640ff40e" />

fake :
<img width="2140" height="1199" alt="fake news" src="https://github.com/user-attachments/assets/26b9ca67-f929-4907-bba2-f1943a08289a" />



---

## 🎯 Key Learnings

* Handling JSON datasets in NLP
* Text preprocessing techniques
* Feature extraction using TF-IDF
* Model training and evaluation
* Deploying ML model using Flask

---

## 🚀 Future Improvements

* 📊 Add confidence score
* 🎯 Improve model accuracy (SVM / Deep Learning)
* 🌐 Deploy app online
* 📱 Mobile-friendly UI

---

⭐ If you like this project, give it a star!
