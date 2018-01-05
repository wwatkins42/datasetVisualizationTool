import plotly.offline as py
import plotly.graph_objs as go
import numpy as np
import argparse
import time
import os
import dateutil.parser

from lib import utils
from lib import dataset
from lib import plotly_utils as pyUtils
from copy import deepcopy

def parseArguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', help='the csv dataset to visualize')
    parser.add_argument('-c', '--cmap', dest='cmap', required=False, choices=pyUtils.cmaps.keys(), default='viridis', help='a custom colormap choice')
    parser.add_argument('-l', '--lines', dest='lines', required=False, action='store_true', default=False, help='set to show lines separating features')
    return parser.parse_args()

def computeHeatmapValues(data, features_type):
    tmp = deepcopy(data)
    heatmap = np.empty_like(data, dtype=float)
    features = []
    for j in range(data.shape[1]):
        column, indices = dataset.dropMissingData(data[:,j], return_indices=True)
        if features_type[j] == dataset.types['missing']:
            features.append(None)
        elif features_type[j] == dataset.types['numerical']:
            column = column.astype(float)
            diff = abs(np.max(column) - np.min(column))
            if diff == 0:
                diff = 1.
            features.append([diff, np.max(column)])
        elif features_type[j] == dataset.types['date']:
            for i in range(tmp.shape[0]):
                date = dateutil.parser.parse(column[i]).timetuple()
                tmp[i,j] = str(time.mktime(date))
            col = tmp[:,j].astype(float)
            diff = abs(np.max(col) - np.min(col))
            features.append([diff, np.max(col)])
        else:
            unique = list(np.unique(column))
            if len(unique) == 1:
                features.append([1, 0])
            else:
                features.append([len(unique)-1, len(unique)-1])
            idxs = np.delete(np.arange(tmp.shape[0]), indices)
            for i in idxs:
                tmp[i,j] = str(unique.index(tmp[i,j]))

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            try:
                point = tmp[i,j].astype(float)
                point = (point - features[j][1]) / features[j][0] + 1.
            except:
                point = 1.001
            heatmap[i,j] = point
    return heatmap

args = parseArguments()
data, labels = dataset.loadCSV(args.dataset)

has_missing_values = False
colors = pyUtils.cmaps[args.cmap]
colorscale = pyUtils.makeColorScale(colors)
missing_color = '#424242'
if dataset.hasMissingValues(data) is True:
    colorscale = pyUtils.appendColorToScale(colorscale, missing_color, p=0.001)
    has_missing_values = True

array_x = np.arange(data.shape[1])
array_y = np.arange(data.shape[0])
array_z, valuesType = dataset.determineValuesType(data, return_keys=True)

missing_count_along_x = data.shape[0] - np.count_nonzero(array_z, axis=0)
missing_count_along_x_range = [np.min(missing_count_along_x), np.max(missing_count_along_x)]
missing_count_along_y = np.count_nonzero(array_z, axis=1)
missing_count_along_y_range = [np.min(missing_count_along_y), np.max(missing_count_along_y)]

features_type = dataset.determineFeaturesType(data)
heatmap_array = computeHeatmapValues(data, features_type)

dlist = [
    #(Heatmap 1 : Features repartition)
    go.Heatmap(
        x=array_x,
        y=array_y,
        z=heatmap_array,
        text=data,
        colorscale=colorscale,
        colorbar=dict(
            x=1.,
            y=0.15,
            len=0.85,
            thicknessmode='fraction',
            thickness=0.025,
            xpad=8,
            ypad=0,
            xanchor='left',
            yanchor='bottom',
            ticks='inside',
            ticklen=5,
            title='Raw values (Feature-relative scale)',
            titleside='right',
            outlinewidth=0.5,
        ),
        hoverinfo="x+y+text",
        xaxis='x',
        yaxis='y',
    ),
    #(Heatmap 2 : Data type)
    go.Heatmap(
        visible=False,
        x=array_x,
        y=array_y,
        z=array_z,
        text=valuesType,
        zmin=0,
        zmax=len(dataset.types.keys())-1+(.01 if has_missing_values else 0),
        colorscale=colorscale,
        colorbar=dict(
            x=1.,
            y=0.15,
            len=0.85,
            thicknessmode='fraction',
            thickness=0.025,
            xpad=8,
            ypad=0,
            xanchor='left',
            yanchor='bottom',
            ticks='inside',
            ticklen=0,
            showticklabels=False,
            title='Values type (Missing, Numerical, String, Date, Boolean)',
            titleside='right',
            outlinewidth=0.5,
        ),
        hoverinfo="text",
        showscale=True,
        xaxis='x',
        yaxis='y',
    ),
    #(Data Completeness plot)
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
        xaxis='x2',
        yaxis='y2',
    ),
    #(xaxis missing values labels)
    go.Scatter(
        x=array_x,
        y=np.zeros_like(array_x),
        text=missing_count_along_x,
        hoverinfo="skip",
        showlegend=False,
        mode='lines',
        xaxis='x3',
        yaxis='y3',
    ),
    #(xaxis missing values labels 2)
    go.Scatter(
        x=array_x,
        y=np.zeros_like(array_x),
        text=missing_count_along_x,
        hoverinfo="skip",
        showlegend=False,
        mode='lines',
        xaxis='x4',
        yaxis='y4',
    ),
]# + pyUtils.makeBoxPlots(data, labels, features_type, axis=0, visible=False, normed=False)


layout = go.Layout(
    # font=dict(family='Balto', size=11, color='#2f2f2f'),
    margin=dict(l=60,r=80,t=90,b=30),

    #(Axes main plot)
    xaxis=dict(
        domain=[0, 0.955],
        tickangle=20,
        ticktext=labels,
        tickvals=array_x,
        ticklen=5,
        showgrid=False,
        zeroline=False,
        side="top",
        fixedrange=True,
    ),
    yaxis=dict(
        title='Indices',
        domain=[0.15, 1],
        tickvals=[0, data.shape[0]-1],
        showgrid=False,
        zeroline=False,
        autorange='reversed'
    ),
    #(Axes Data Completeness)
    xaxis2=dict(
        title='Data<br>Completeness',
        titlefont=dict(size=11),
        tickfont=dict(size=11),
        domain=[0.96, 1],
        range=[missing_count_along_y_range[0]-0.05, missing_count_along_y_range[1]+0.05],
        ticklen=3,
        tickwidth=3,
        tickvals=missing_count_along_y_range,
        showgrid=False,
        zeroline=False,
        fixedrange=True,
        side="top",
    ),
    yaxis2=dict(
        autorange=False,
        range=[data.shape[0]-0.5, -0.5],
        showgrid=False,
        showticklabels=False,
        zeroline=False,
        overlaying='y',
    ),
    #(Axes missing values labels)
    xaxis3=dict(
        tickfont=dict(size=10,color='#505050'),
        range=[-0.5, data.shape[1]-0.5],
        tickvals=np.arange(len(missing_count_along_x))[0::2],
        ticktext=missing_count_along_x[0::2],
        ticklen=1,
        tickwidth=3,
        showgrid=False,
        zeroline=False,
        fixedrange=True,
        overlaying='x',
    ),
    yaxis3=dict(
        domain=[0.13, 0.15],
        range=[1, 2],
        showticklabels=False,
        showgrid=False,
        zeroline=False,
        fixedrange=True,
    ),
    #(Axes missing values labels 2)
    xaxis4=dict(
        title='Missing values (per feature)',
        titlefont=dict(size=11,color='#505050',),
        tickfont=dict(size=10,color='#505050'),
        tickcolor='#b0b0b0',
        range=[-0.5, data.shape[1]-0.5],
        tickvals=np.arange(len(missing_count_along_x))[1::2],
        ticktext=missing_count_along_x[1::2],
        ticklen=15,
        showgrid=False,
        zeroline=False,
        fixedrange=True,
        overlaying='x',
    ),
    yaxis4=dict(
        domain=[0.12, 0.13],
        range=[1, 2],
        showticklabels=False,
        showgrid=False,
        zeroline=False,
        fixedrange=True,
    ),

    #(Title, which is the name of the file)
    # annotations=[
    #     dict(
    #         x=0.5,
    #         y=-0.02,
    #         font=dict(size=14),
    #         xref='paper',
    #         yref='paper',
    #         text=os.path.basename(args.dataset),
    #         xanchor='middle',
    #         showarrow=False
    #     )
    # ]
    # boxgroupgap=(4./data.shape[1]), # -> rescales the box plots in function of the number of features
)
#(Lines on main plot)
shapes = pyUtils.makeVerticalLines(np.arange(len(array_x)+1)-0.5, y0=0.15, xref='x', yref='paper', color='#000000', linewidth=0.5) + \
         pyUtils.makeHorizontalLines([0.15,1], x0=0., x1=0.955, xref='paper', yref='paper', color='#000000', linewidth=0.5)

#(Dropdown menu for color map choices)
buttons_cmaps=dict(
    buttons=pyUtils.makeColorscaleButtons(pyUtils.cmaps, add_missing=missing_color if has_missing_values else None),
    type='dropdown',
    direction='left',
    active=(pyUtils.cmaps.keys().index(args.cmap)),
    pad={'t':7},
    showactive=True,
    x=.96,
    y=0.04,
    xanchor='left',
    yanchor='top',
    borderwidth=0.5,
)
#(Buttons to show/hide lines)
buttons_lines=dict(
    buttons=[
        dict(
            args=['shapes', []],
            label='Hide',
            method='relayout'
        ),
        dict(
            args=['shapes', shapes],
            label='Show',
            method='relayout'
        )
    ],
    type='buttons',
    direction='right',
    active=(1 if args.lines else 0),
    showactive=True,
    pad={'t':8},
    x=.96,
    y=0.15,
    xanchor='left',
    yanchor='top',
    borderwidth=0.5
)
#(Dropdown menu to select the plot to display)
buttons_plot=dict(
    buttons=[
        dict(
            args=['visible', [True, False, True]],
            label='Features repartition',
            method='restyle'
        ),
        dict(
            args=['visible', [False, True, True]],
            label='Data type',
            method='restyle'
        )
    ],
    type='dropdown',
    direction='left',
    active=0,
    showactive=True,
    pad={'t':4},
    x=0.96,
    y=0.09,
    xanchor='left',
    yanchor='top',
    borderwidth=0.5
)
layout['updatemenus'] = [buttons_cmaps, buttons_lines, buttons_plot]

if args.lines is True:
    layout['shapes'] = shapes

fig = go.Figure(data=dlist, layout=layout)
py.plot(fig, filename='csv-plot.html', show_link=False)
