from flask import Flask
from flask import render_template
from flask import request
from inference import get_answer
from transformers import AutoTokenizer, AutoModelForCausalLM

app = Flask(__name__)
model = AutoModelForCausalLM.from_pretrained("training/training-one")
tokenizer = AutoTokenizer.from_pretrained("training/training-one")


@app.route("/")
def hello():
    return render_template("index.html")


@app.route("/answer", methods=["POST"])
def request_answer():
    question = request.form["question"]

    answer = get_answer(question, model, tokenizer)
    return answer
