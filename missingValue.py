import plotly.offline as py
import plotly.graph_objs as go
import numpy as np
import re
import argparse
from lib import dataset
from lib import plotly_utils as pyUtils

def parseArguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', help='the csv dataset to visualize')
    parser.add_argument('-c', '--cmap', dest='cmap', required=False, choices=pyUtils.cmaps.keys(), default='gold', help='a custom colormap choice')
    parser.add_argument('-l', '--lines', dest='lines', required=False, action='store_true', default=False, help='set to show lines')
    return parser.parse_args()

def determineFeatureType(data):
    pattern = {
        'numerical':re.compile(r"[-+]?\d*\.*\d+"),
        'date':re.compile(r"\d+[-]\d+[-]\d+")
    }
    types = {'missing':0, 'numerical':1, 'string':2, 'date':3}
    tmp = np.empty_like(data, dtype=int)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if data[i,j] == '' or data[i,j] == 'nan':
                tmp[i,j] = types['missing']
            elif pattern['date'].match(data[i,j]):
                tmp[i,j] = types['date']
            elif pattern['numerical'].match(data[i,j]):
                tmp[i,j] = types['numerical']
            else:
                tmp[i,j] = types['string']
    return tmp, len(types)


args = parseArguments()
colors = pyUtils.cmaps[args.cmap]
colorscale = pyUtils.makeColorScale(colors)

data, labels = dataset.loadCSV(args.dataset)

array_x = np.arange(data.shape[1])
array_y = np.arange(data.shape[0])
array_z, zmax = determineFeatureType(data)

missing_count_along_x = data.shape[0] - np.count_nonzero(array_z, axis=0)
missing_count_along_y = np.count_nonzero(array_z, axis=1)
missing_count_along_y_range = [np.min(missing_count_along_y), np.max(missing_count_along_y)]

dlist = [
    go.Heatmap(
        x=array_x,
        y=array_y,
        z=array_z,
        zmin=0,
        zmax=zmax-1,
        colorscale=colorscale,
        hoverinfo="x+y",
        showscale=False,
        xaxis='x',
        yaxis='y',
    ),
    go.Bar(
        x=array_x,
        y=missing_count_along_x,
        text=labels,
        hoverinfo="y",
        showlegend=False,
        marker=dict(
            color='#ffffff',
            line={
                'color':"#000000",
                'width':0.5
            },
        ),
        width=1,
        xaxis='x2',
        yaxis='y2',
    ),
    go.Scatter(
        x=missing_count_along_y,
        y=array_y,
        hoverinfo="x+y",
        showlegend=False,
        line={
            'color':"#000000",
            'shape':'vhv',#should be hvh...
            'width':1
        },
        mode='lines',
        xaxis='x3',
        yaxis='y3',
    ),
] + pyUtils.makeColorLegend(['missing', 'numerical', 'string', 'date'], colors)


layout = go.Layout(
    margin=dict(
        l=50,
        r=80,
        t=90,
        b=30
    ),
    legend=dict(
        x=-0.015,
        y=-0.075,
        orientation="h"
    ),
    xaxis=dict(
        domain=[0, 0.955],
        tickangle=20,
        ticktext=labels,
        tickvals=array_x,
        ticklen=5,
        showgrid=False,
        zeroline=False,
        side="top",
    ),
    yaxis=dict(
        title='Indices',
        domain=[0.05, 1],
        tickvals=[0, data.shape[0]-1],
        showgrid=False,
        zeroline=False,
        autorange='reversed'
    ),
    xaxis2=dict(
        domain=[0, 0.955],
        showgrid=False,
        tickvals=array_x,
        ticklen=28,
        ticks='outside',
        zeroline=False,
        overlaying='x',
    ),
    yaxis2=dict(
        domain=[0, 0.04],
        showgrid=False,
        showticklabels=False,
        zeroline=False,
        fixedrange=True,
        autorange='reversed'
    ),
    xaxis3=dict(
        title='Data<br>Completeness',
        titlefont=dict(size=11),
        tickfont=dict(size=11),
        domain=[0.96, 1],
        range=[np.min(missing_count_along_y)-0.05, np.max(missing_count_along_y)+0.05],
        ticklen=3,
        tickwidth=3,
        tickvals=missing_count_along_y_range,
        showgrid=False,
        zeroline=False,
        fixedrange=True,
        side="top",
    ),
    yaxis3=dict(
        autorange=False,
        range=[data.shape[0]-0.5, 0.5],
        showgrid=False,
        showticklabels=False,
        zeroline=False,
        overlaying='y',
    ),
)
if args.lines is True:
    layout['shapes'] = pyUtils.makeVerticalLines(array_x-0.5, y0=0.05, color='#000000', linewidth=0.5)

fig = go.Figure(data=dlist, layout=layout)
py.plot(fig, filename='csv-plot.html')
