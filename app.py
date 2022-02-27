from flask import Flask, redirect, url_for, render_template, request, flash
# from bs4 import BeautifulSoup

import validate
import analysis


app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/input', methods=['POST', 'GET'])
def input():
    if request.method == "POST":

        rc = float(request.form["relative-compactness"]) / 100
        width = float(request.form["width"])
        length = float(request.form["length"])
        height = float(request.form["overall-height"])
        ori = float(request.form["orientation"])
        windows = float(request.form["number-of-windows"])

        # ga = float(request.form["glazing-area"])
        # gad = float(request.form["glazing-area-distribution"])
        # sa = float(request.form["surface-area"])
        # wa = float(request.form["wall-area"])
        # ra = float(request.form["roof-area"])
        
        window_area = 3.0

        sa = 2 * width * length + 2 * length * height + 2 * height * width
        wa = 2 * width * height + 2 * length * height
        ra = width * length
        oh = height

        ga = windows * window_area / wa
        

        # ga = windows * windowArea / 100
        # gad =

        # result_heating, result_cooling = validate.get_prediction(rc, sa, wa, ra, oh, ori, ga, gad)

        result_cooling, result_heating = validate.get_prediction(rc, sa, wa, ra, oh, ori, ga, gad)



        return redirect(url_for("user", usr=result_heating))
    else:
        return render_template('input.html')

@app.route("/<usr>")
def user(usr):
    return f"<h1>{usr}</h1>"


@app.route('/show_graph')
def get_graph():
    graphs = analysis.get_plots()

    graph_1 = graphs[0]
    graph_2 = graphs[1]
    graph_3 = graphs[2]
    graph_4 = graphs[3]
    graph_5 = graphs[4]
    graph_6 = graphs[5]
    graph_7 = graphs[6]
    graph_8 = graphs[7]

    return render_template('showGraph.html', graph_1=graphs[0], graph_2=graphs[1], graph_3=graphs[2], graph_4=graphs[3], graph_5=graphs[4], graph_6=graphs[5], graph_7=graphs[6], graph_8=graphs[7])


    # return redirect(url_for("show_graphs", graphs=graphs))

# @app.route('/show_graphs')
# def show_graphs(graphs):
#     graph_1 = graphs[0]
#     graph_2 = graphs[1]
#     graph_3 = graphs[2]
#     graph_4 = graphs[3]
#     graph_5 = graphs[4]
#     graph_6 = graphs[5]
#     graph_7 = graphs[6]
#     graph_8 = graphs[7]


