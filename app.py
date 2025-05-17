import pandas as pd
from flask import Flask, request, jsonify, render_template
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import joblib

# Load model and vectorizer (after training once)
model = joblib.load("spam_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    message = request.form["message"]
    vec = vectorizer.transform([message])
    prediction = model.predict(vec)
    result = "Spam" if prediction[0] == 1 else "Ham (Not Spam)"
    return render_template("index.html", message=message, result=result)

if __name__ == "__main__":
    app.run(debug=True)
