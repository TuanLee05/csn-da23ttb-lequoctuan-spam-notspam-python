from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Load mô hình
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    email_text = request.form.get("email_text")

    if not email_text or email_text.strip() == "":
        return render_template(
            "result.html",
            label="⚠️ Vui lòng nhập nội dung email"
        )

    X = vectorizer.transform([email_text])
    proba = model.predict_proba(X)[0]

    spam_index = list(model.classes_).index("spam")
    spam_prob = proba[spam_index]

    if spam_prob > 0.6:
        label = "🚫 SPAM"
    elif spam_prob < 0.4:
        label = "✅ HAM (không spam)"


    return render_template("result.html", label=label)

if __name__ == "__main__":
    app.run(debug=True)
