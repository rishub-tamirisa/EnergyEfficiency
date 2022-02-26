from flask import Flask, redirect, url_for, render_template, request, flash
import validate

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/input', methods=['POST', 'GET'])
def input():
    if request.method == "POST":
        rc = float(request.form["relative-compactness"]) / 100
        sa = float(request.form["surface-area"])
        wa = float(request.form["wall-area"])
        ra = float(request.form["roof-area"])
        oh = float(request.form["overall-height"])
        ori = float(request.form["orientation"])
        ga = float(request.form["glazing-area"])
        gad = float(request.form["glazing-area-distribution"])

        # result_heating = validate.get_prediction(rc, sa, wa, ra, oh, ori, ga, gad)
        result_heating = rc

        return redirect(url_for("user", usr=result_heating))
    else:
        return render_template('input.html')

@app.route("/<usr>")
def user(usr):
    return f"<h1>{usr}</h1>"
