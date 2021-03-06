from flask import Flask, redirect, url_for, render_template, request, flash
# from bs4 import BeautifulSoup

import validate
import analysis


app = Flask(__name__)



width = 0
height = 0
length = 0

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
        gad = float(request.form["distribution"])
        # ga = float(request.form["glazing-area"])
        # gad = float(request.form["glazing-area-distribution"])
        # sa = float(request.form["surface-area"])
        # wa = float(request.form["wall-area"])
        # ra = float(request.form["roof-area"])

        window_area = 3.0

        # window distribution

        sa = 2 * width * length + 2 * length * height + 2 * height * width
        wa = 2 * width * height + 2 * length * height
        ra = width * length
        oh = height

        ga = windows * window_area / wa

        # ga = windows * windowArea / 100
        # gad =

        # result_heating, result_cooling = validate.get_prediction(rc, sa, wa, ra, oh, ori, ga, gad)

        result_heating, result_cooling = validate.get_prediction(
            rc, sa, wa, ra, oh, ori, ga, gad)

        # result_heating = [rc, sa, wa, ra, oh, ori, ga, gad]

        return redirect(url_for("result", heat=result_heating, cool=result_cooling, w=width, l=length, h=height))
    else:
        return render_template('input.html')


@app.route("/<heat>/<cool>/<w>/<l>/<h>")
def result(heat, cool, w, l, h):
    return render_template('result.html', heat=heat, cool=cool, w=w, l=l, h=h)


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


def run_app():
    app.run()