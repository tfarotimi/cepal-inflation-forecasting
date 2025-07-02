# This file contains functions for analyzing the residuals of a forecast model.
# Last update: 08.14.2023 by Inflation Forecasting Farotimi

from statsmodels.graphics.gofplots import qqplot 
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def analyze_residuals(result):
    '''
    Analyze residuals of forecast model and plot results.

    Parameters
    ----------
    result : dict
        Dictionary containing the results of the forecast model.

    Returns
    -------
    None.
    
    '''
    
    residuals = result.meta['residuals']
  
    #plot residuals and summary statistics 
    fig = make_subplots(rows=1, cols=3, subplot_titles=("Residuals", "Histogram", "QQ Plot"))
    fig.add_trace(go.Line(x = residuals.index, y=residuals), row=1, col=1)
    fig.add_trace(go.Histogram(x=residuals), row=1, col=2)

    #get qqplot data
    qqplot_data = qqplot(residuals, line='s').gca().lines

    #add qqplot data to plot 
    fig.add_trace({
        'type': 'scatter',
        'x': qqplot_data[0].get_xdata(),
        'y': qqplot_data[0].get_ydata(),
        'mode': 'markers',
        'marker': {
            'color': '#19d3f3'
        },
        'name': 'Data',
        'showlegend': False,
    
    },row=1, col=3)

    #add qqplot data to plot
    fig.add_trace({
        'type': 'scatter',
        'x': qqplot_data[1].get_xdata(),
        'y': qqplot_data[1].get_ydata(),
        'mode': 'lines',
        'line': {
            'color': '#636efa'
        }

    },row=1, col=3)

    #add training data to plot
    fig.add_shape(type="line",x0=result.index[int(0.8*len(result.index))],y0=0,x1=result.index[int(0.8*len(result.index))],y1=1,row=1,col=1,line=dict(color="RoyalBlue",width=1,dash="dashdot",))
    
    #add summary statistics to plot
    fig.add_annotation(x=0.5, y=0.95, xref="paper", yref="paper",text=round(residuals.describe(),2).to_string(), bordercolor="black", showarrow=False,font=dict(size=10),bgcolor="white",borderwidth=1)

    #update layout
    fig['layout'].update({
        'title': 'Residual Analysis for ' + result.meta['name'],
        'showlegend': False
    })


    fig.show()  

