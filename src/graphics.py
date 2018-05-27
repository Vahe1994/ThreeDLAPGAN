import plotly.plotly as py
import plotly.graph_objs as go
from  plotly.offline import iplot,init_notebook_mode
import numpy as np
from sklearn.neighbors import NearestNeighbors

# plotly.tools.set_credentials_file(username='', api_key='')
def get_plot(pclouds=None,id_of_obj = 0,name = 'chair',mode=False):
    init_notebook_mode(mode)
    x, y, z = pclouds[id_of_obj,:,0],pclouds[id_of_obj,:,1],pclouds[id_of_obj,:,2]
    trace1 = go.Scatter3d(
        x=x,
        y=y,
        z=z,
        mode='markers',
        marker=dict(
            size=4,
            line=dict(
                color='rgba(217, 217, 217, 0.14)',
                width=0.5
            ),
            opacity=0.8
        )
    )


    data = [trace1]
    layout = go.Layout(
        margin=dict(
            l=0,
            r=0,
            b=0,
            t=0
        )
    )
    fig = go.Figure(data=data, layout=layout)
    if mode:
        return py.iplot(fig, filename= name)
    else:
        return iplot(fig, filename= name)