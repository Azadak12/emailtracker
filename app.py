import os
import pandas as pd
from flask import Flask, request, jsonify, render_template
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import joblib

# Load the pre-trained model and vectorizer
model = joblib.load("spam_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Initialize the Flask app
app = Flask(__name__)

# Home route
@app.route("/")
def home():
    return render_template("index.html")

# Predict route
@app.route("/predict", methods=["POST"])
def predict():
    message = request.form["message"]
    vec = vectorizer.transform([message])
    prediction = model.predict(vec)
    result = "Spam" if prediction[0] == 1 else "Ham (Not Spam)"
    return render_template("index.html", message=message, result=result)

# Main entry point
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Default to port 5000
    app.run(host="0.0.0.0", port=port, debug=True)
