from flask import Flask
from flask import render_template
from flask import request
from inference import get_answer
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL = None
TOKENIZER = None
app = Flask(__name__)


@app.route("/")
def hello():
    return render_template("index.html")


@app.route("/answer", methods=["POST"])
def request_answer():
    question = request.form["question"]

    if MODEL is None or TOKENIZER is None:
        return "Model not loaded"

    answer = get_answer(question, MODEL, TOKENIZER)
    return answer


if __name__ == "__main__":
    MODEL = AutoModelForCausalLM.from_pretrained("training/model")
    TOKENIZER = AutoTokenizer.from_pretrained("training/model")

    app.run(debug=False)
