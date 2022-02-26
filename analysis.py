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



vars = ['X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8']
yvars = ['Y1', 'Y2']

# print(x[['X1']])
# print(y[['Y1']])
# heat = [x[['X1']], y[['Y1']]]
# cool = [x[['X1']], y[['Y2']]]

# heat = pd.concat(heat, axis=1)
# cool = pd.concat(cool, axis=1)

# print(result)



# heating = go.Scatter(
#     x=x[['X1']], y=y[['Y1']], name='Relative Compactness Effect on Heating Efficiency', yaxis='y2'
# )
# fig = go.Figure(data=heating)
# fig.show()
# cooling = go.Scatter(
#     x=x[['X1']], y=y[['Y2']], name='Relative Compactness Effect on Cooling Efficiency', yaxis='y2'
# )

# fig = make_subplots(specs=[[{"secondary_y": True}]])
# fig.add_trace(heating)
# fig.add_trace(cooling,secondary_y=True)
# # fig['layout'].update(height = 600, width = 800, title = title,xaxis=dict(
# #       tickangle=-90
# #     ))
# fig.show()
html_plots = list()

for id in vars:
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
    fig = sp.make_subplots(rows=1, cols=2,  subplot_titles=(id + ' Effect on Heating Efficiency',  id + ' Effect on Cooling Efficiency')) 

    # Get the Express fig broken down as traces and add the traces to the proper plot within in the subplot
    for traces in figure1_traces:
        fig.append_trace(traces, row=1, col=1)
    for traces in figure2_traces:
        fig.append_trace(traces, row=1, col=2)

    fig.show()

    html_plots.append(plot(fig, include_plotlyjs=False, output_type='div'))

# print(html_plots[1])
# html_plots = list()



# buffer = io.StringIO()

# fig = heating
# fig.write_html(buffer)

# html_bytes = buffer.getvalue().encode()
# encoded = b64encode(html_bytes).decode()

# graph = dash.Dash(__name__)
# graph.layout = html.Div([
#     dcc.Graph(id="graph", figure=fig),
#     html.A(
#         html.Button("Download HTML"), 
#         id="download",
#         href="data:text/html;base64," + encoded,
#         download="plotly_graph.html"
#     )
# ])

# graph.run_server(debug=True)

# heating.show()
# print(x[['X4']])
# print(len(x))
# t = linspace(0, len(x), num=len(x))

# plt.plot(x[['X1']], y[['Y1']])
# plt.scatter(x[['X1']], y[['Y1']])
# plt.show()
# combine = dict(zip(x.X1, y.Y1))
# print (combine)
# plotly.plot(Scatter()