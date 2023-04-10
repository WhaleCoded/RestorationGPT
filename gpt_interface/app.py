from flask import Flask
from flask import render_template
from flask import request
from inference import get_answer
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL = None
TOKENIZER = None
MODEL_VERSION = 1
app = Flask(__name__)


@app.route("/")
def hello():
    return render_template("index.html", model_version=MODEL_VERSION)


@app.route("/answer", methods=["POST"])
def request_answer():
    question = request.form["question"]

    if MODEL is None or TOKENIZER is None:
        return "Model not loaded"

    answer = get_answer(question, MODEL, TOKENIZER, MODEL_VERSION)
    return answer


if __name__ == "__main__":
    import os

    # Get model version from environment variable
    model_version = int(os.environ.get("MODEL_VERSION", 1))
    model_version_text = "version_" + str(model_version)
    MODEL_VERSION = int(model_version)

    print("Loading model " + model_version_text)
    MODEL = AutoModelForCausalLM.from_pretrained("training/" + model_version_text)
    TOKENIZER = AutoTokenizer.from_pretrained("training/" + model_version_text)
    print("Model loaded")

    port = 5000
    if model_version == 2:
        port = 5001
    elif model_version == 3:
        port = 5002

    app.title = "RestorationGPT V_" + str(model_version)
    app.run(
        debug=False,
        port=port,
    )
