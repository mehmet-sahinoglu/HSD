from flask import Flask, render_template, request
from transformers import pipeline

app = Flask(__name__)

# Model sadece bir kez yüklenir
_hate_classifier = pipeline(
    "text-classification",
    model="unitary/toxic-bert",
    tokenizer="unitary/toxic-bert",
    return_all_scores=True
)

def detect_hate_speech(text: str) -> dict:
    result = _hate_classifier(text)[0]
    sorted_res = sorted(result, key=lambda x: x["score"], reverse=True)
    top_label = sorted_res[0]
    return {
        "label": top_label["label"],
        "confidence": round(top_label["score"] * 100)
    }

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        user_text = request.form.get("user_text")
        result = detect_hate_speech(user_text)
        return render_template("result_hatespeech.html", text=user_text, result=result)
    return render_template("home_hatespeech.html")

if __name__ == "__main__":
    app.run()
