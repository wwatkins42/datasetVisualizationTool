import plotly.offline as py
import plotly.graph_objs as go
import plotly.figure_factory as ff
import numpy as np
import argparse
import time
import os
import dateutil.parser

from lib import dataset
from lib import plotly_utils as pyUtils
from copy import deepcopy

def parseArguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', help='the csv dataset to visualize')
    parser.add_argument('-c', '--cmap', dest='cmap', required=False, choices=pyUtils.cmaps.keys(), default='viridis', help='a custom colormap choice')
    parser.add_argument('-l', '--lines', dest='lines', required=False, action='store_true', default=False, help='set to show lines separating features')
    parser.add_argument('-m', '--missing-labels', dest='missing_labels', nargs='+', default=['','nan','NaN','n/a','N/A'], required=False, help='the missing labels of the dataset')
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
            # string (TODO : could have a sort by ascii order for the strings)
            unique = list(np.unique(column))
            if len(unique) == 1:
                features.append([1, 0])
            else:
                features.append([len(unique)-1, len(unique)-1])
            idxs = np.delete(np.arange(tmp.shape[0]), indices)
            for i in idxs:
                tmp[i,j] = str(unique.index(tmp[i,j]))
        # TODO : implement boolean

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

missing_count_along_y = np.count_nonzero(array_z, axis=1)
missing_count_along_y_range = [np.min(missing_count_along_y), np.max(missing_count_along_y)]

features_type = dataset.determineFeaturesType(data)
heatmap_array = computeHeatmapValues(data, features_type)

dlist = [
    go.Heatmap( # Features repartition informations
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
            title='raw values (Feature-relative scale)',
            titleside='right',
            outlinewidth=0.5,
        ),
        hoverinfo="x+y+text",
        xaxis='x',
        yaxis='y',
    ),
    go.Heatmap( # Data type informations
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
            ticklen=5,
            # dtick=0.1,
            title='values type',
            titleside='right',
            outlinewidth=0.5,
        ),
        hoverinfo="text",
        showscale=True,
        xaxis='x',
        yaxis='y',
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
]# + pyUtils.makeBoxPlots(data, labels, features_type, axis=2, normed=True)


layout = go.Layout(
    # font=dict(family='Balto', size=11, color='#2f2f2f'),
    margin=dict(
        l=60,
        r=80,
        t=90,
        b=30
    ),
    legend=dict(
        x=-0.015,
        y=-0.075,
        orientation="h",
        traceorder='normal'
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
    # xaxis2=dict(
    #     domain=[0, 0.955],
    #     showgrid=False,
    #     ticks=False,
    #     showticklabels=False,
    #     zeroline=False,
    #     overlaying='x',
    #     fixedrange=True,
    # ),
    # yaxis2=dict(
    #     domain=[0, 0.15],
    #     showgrid=False,
    #     showticklabels=False,
    #     zeroline=False,
    #     fixedrange=True,
    #     autorange='reversed'
    # ),
    xaxis3=dict(
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
    yaxis3=dict(
        autorange=False,
        range=[data.shape[0]-0.5, -0.5],
        showgrid=False,
        showticklabels=False,
        zeroline=False,
        overlaying='y',
    ),
    # boxgroupgap=(4./data.shape[1]), # -> rescales the box plots in function of the number of features
    annotations=[
        dict(
            x=0.5,
            y=-0.04,
            font=dict(size=14),
            xref='paper',
            yref='paper',
            text=os.path.basename(args.dataset),
            xanchor='middle',
            showarrow=False
        )
    ]
    # annotations = pyUtils.makeHorizontallyAlignedAnnotations(
    #     array_x,
    #     array_x,
    #     # y=-0.03,
    #     y=0.115,
    #     xref='x2',
    #     yref='paper',
    #     xanchor='middle'
    # )
)
shapes = pyUtils.makeVerticalLines(np.arange(len(array_x)+1)-0.5, y0=0.15, xref='x', yref='paper', color='#000000', linewidth=0.5) + \
         pyUtils.makeHorizontalLines([0.15,1], x0=0., x1=0.955, xref='paper', yref='paper', color='#000000', linewidth=0.5)

buttons_cmaps=dict(
    buttons=pyUtils.makeColorscaleButtons(pyUtils.cmaps, add_missing=missing_color if has_missing_values else None),
    type='dropdown',
    direction='left',
    active=(pyUtils.cmaps.keys().index(args.cmap)),
    pad={'r':4, 't':4},
    showactive=False,
    x=.965,
    y=0.15,
    xanchor='left',
    yanchor='top',
    borderwidth=0.5,
)
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
    pad={'r':4, 't':4},
    x=.965,
    y=0.04,
    xanchor='left',
    yanchor='top',
    borderwidth=0.5
)
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
    x=.965,
    y=0.09,
    xanchor='left',
    yanchor='top',
    borderwidth=0.5
)

layout['updatemenus'] = [buttons_cmaps, buttons_lines, buttons_plot]

if args.lines is True:
    layout['shapes'] = shapes

fig = go.Figure(data=dlist, layout=layout)
py.plot(fig, filename='csv-plot.html')