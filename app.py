from flask import Flask, redirect, url_for, render_template, request, flash

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/input', methods=['POST', 'GET'])
def input():
    if request.method == "POST":
        rc = request.form["relative-compactness"]
        sa = request.form["surface-area"]
        wa = request.form["wall-area"]
        ra = request.form["roof-area"]
        oh = request.form["overall-height"]
        ori = request.form["orientation"]
        ga = request.form["glazing-area"]
        gad = request.form["glazing-area-distribution"]
        dict = {'X1':[rc],
            'X2':[sa],
            'X3':[wa],
            'X4':[ra],
            'X5':[oh],
            'X6':[ori],
            'X7':[ga],
            'X8':[gad],
        }
        return redirect(url_for("user", usr=dict))
    else:     
        return render_template('input.html')

@app.route("/<usr>")
def user(usr):
    return f"<h1>{usr}</h1>"