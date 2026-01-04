from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Load mô hình và vectorizer
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    email_text = request.form["email_text"]
    X = vectorizer.transform([email_text])
    prediction = model.predict(X)[0]
    label = "SPAM" if prediction == "spam" else "HAM (không spam)"
    return render_template("result.html", email=email_text, label=label)

if __name__ == "__main__":
    app.run(debug=True)
