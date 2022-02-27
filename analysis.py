from turtle import color
from matplotlib.axis import YAxis
import sklearn
import data
from numpy import *
import math
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import plotly.subplots as sp
from plotly.offline import plot
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pickle



x = data.Data.x
y = data.Data.y


labels = ['Relative Compactness',
        'Surface Area',
        'Wall Area',
        'Roof Area',
        'Overall Height',
        'Orientation',
        'Glazing Area',
        'Glazing Area Distribution']

vars = ['X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8']
yvars = ['Y1', 'Y2']

#array containing html for all 8 scatter plots


        

def get_plots():
    html_plots = list()
    for i, id in enumerate(vars):

        heat = [x[[id]], y[['Y1']]]
        cool = [x[[id]], y[['Y2']]]
        heat = pd.concat(heat, axis=1)
        cool = pd.concat(cool, axis=1)
        
        hstddev = np.std(np.asarray(y[['Y1']]))
        hmean = np.mean(np.asarray(y[['Y1']]))
        hcoeff = hstddev / hmean

        xstddev = np.std(np.asarray(x[[id]]))
        xmean = np.mean(np.asarray(x[[id]]))
        xcoeff = xstddev / xmean

        cstddev = np.std(np.asarray(y[['Y2']]))
        cmean = np.mean(np.asarray(y[['Y2']]))
        ccoeff = cstddev / cmean

        # print(heat)
        # print (hstddev)
        # print (mean)
        heating = px.scatter(heat, x=id, y='Y1', trendline='lowess')

        cooling = px.scatter(cool, x=id, y='Y2', trendline='lowess')

        figure1_traces = []
        figure2_traces = []
        for trace in range(len(heating["data"])):
            heating["data"][trace]['marker']['color'] = '#ff295e'
            figure1_traces.append(heating["data"][trace])
        for trace in range(len(cooling["data"])):
            cooling["data"][trace]['marker']['color'] = '#009dff'
            figure2_traces.append(cooling["data"][trace])

        #Create a 1x2 subplot
        fig = sp.make_subplots(rows=1, cols=2,  subplot_titles=(labels[i] + ' Effect on Heating Load',  labels[i] + ' Effect on Cooling Load'))
        fig['layout']['xaxis']['title']=labels[i] + " (Co. of Var: " + str(np.round(xcoeff, 2)) + ")"
        fig['layout']['xaxis2']['title']=labels[i] + " (Co. of Var: " + str(np.round(xcoeff, 2)) + ")"
        fig['layout']['yaxis']['title']='Heating Efficiency' + " (Co. of Var: " + str(np.round(hcoeff, 2)) + ")"
        fig['layout']['yaxis2']['title']='Cooling Efficiency' + " (Co. of Var: " + str(np.round(ccoeff, 2)) + ")"
        # Get the Express fig broken down as traces and add the traces to the proper plot within in the subplot

        for traces in figure1_traces:
            fig.append_trace(traces, row=1, col=1)

        for traces in figure2_traces:
            fig.append_trace(traces, row=1, col=2)

        fig.update_traces(marker=dict(size=5,
                                    opacity=0.2,
                                ),
                    selector=dict(mode='markers'))

        fig.update_layout(
            autosize=True,
            width=1000,
            height=500,

        )

        # fig.show()

        html_plots.append(plot(fig, include_plotlyjs=True, output_type='div'))
    return html_plots
# print(len(get_plots()))
# get_plots()
