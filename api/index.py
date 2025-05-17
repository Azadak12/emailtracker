from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)
model = joblib.load("spam_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

@app.route('/', methods=['GET'])
def home():
    return "Spam Detector API is running!"

@app.route('/check', methods=['POST'])
def check_spam():
    data = request.get_json()
    message = data.get("message", "")
    vec = vectorizer.transform([message])
    result = model.predict(vec)[0]
    return jsonify({"result": "Spam" if result == 1 else "Ham (Not Spam)"})

# Needed for Vercel
# Make sure Vercel knows this is the handler
def handler(request, context=None):
    return app(request.environ, start_response)

# Flask compatibility fix for Vercel
from werkzeug.wrappers import Request, Response

def start_response(status, headers):
    def wrapper(body):
        return Response(body, status=int(status.split()[0]), headers=dict(headers))
    return wrapper
