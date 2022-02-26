from matplotlib.axis import YAxis
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

        heating = px.scatter(heat, x=id, y='Y1')
        cooling = px.scatter(cool, x=id, y='Y2')

        figure1_traces = []
        figure2_traces = []
        for trace in range(len(heating["data"])):
            figure1_traces.append(heating["data"][trace])
        for trace in range(len(cooling["data"])):
            figure2_traces.append(cooling["data"][trace])

        #Create a 1x2 subplot
        fig = sp.make_subplots(rows=1, cols=2,  subplot_titles=(labels[i] + ' Effect on Heating Efficiency',  labels[i] + ' Effect on Cooling Efficiency'))
        fig['layout']['xaxis']['title']=labels[i]
        fig['layout']['xaxis2']['title']=labels[i]
        fig['layout']['yaxis']['title']='Heating Efficiency'
        fig['layout']['yaxis2']['title']='Cooling Efficiency'
        # Get the Express fig broken down as traces and add the traces to the proper plot within in the subplot
        for traces in figure1_traces:
            fig.append_trace(traces, row=1, col=1)
        for traces in figure2_traces:
            fig.append_trace(traces, row=1, col=2)

        fig.update_traces(marker=dict(size=5,
                                    opacity=0.5,
                                line=dict(width=0,
                                            color='yellow')),
                    selector=dict(mode='markers'))

        fig.update_layout(
            autosize=True,
            width=1000,
            height=500,

        )


        html_plots.append(plot(fig, include_plotlyjs=True, output_type='div'))
    return html_plots
# print(len(get_plots()))
get_plots()
