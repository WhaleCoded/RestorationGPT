from flask import Flask
from flask import render_template
from flask import request

app = Flask(__name__)


@app.route("/")
def hello():
    return render_template("index.html")


@app.route("/answer", methods=["POST"])
def request_answer():
    print("We made it to the server!")
    question = request.form["question"]

    print(question)
    return "The answer to {} is 42".format(question)
