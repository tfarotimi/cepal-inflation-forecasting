# This file contains functions for creating a dashboard to compare model results and performance. 
# Last update: 08.14.2023 by Inflation Forecasting Farotimi

# import libraries
import pandas as pd
import numpy as np

#plotly
import plotly.graph_objs as go
import plotly.offline as pyo
from plotly.subplots import make_subplots
pyo.init_notebook_mode()

#scikit-learn
from sklearn.metrics import mean_squared_error,r2_score

# compare model results and performance
def model_compare(configs): 
    ''' 
    Given a list of config ojbects,  this function will compare the performance of each model on a grid of plots.
    The function will return a list of results for each model.

    Parameters
    ----------
    configs : list of config objects
        A list of config objects that contain the model, data, country, target, params for each model.

    Returns
    -------
    results : list of results
        A list of results for each model.


    '''
    #set number of rows and columns for grid of plots
    if len(configs) > 3:
        cols = 3
        rows = 1 + len(configs)//3
    else:
        cols = len(configs)
        rows = 1

    #run each model and add it to a list of results
    results = []
    for config in configs:
        model = config['model']
        result = model(data = config['data'],country = config['country'], target = config['target'], params = config['params'], lags = config['lags'],predictors = config['predictors'],walk_forward = config['walk_forward'])
        results.append(result)

    #show plot of each model in a grid
    fig = make_subplots(rows=rows, 
                        cols=cols, 
                        vertical_spacing=0.2,
                        print_grid=True,
                        subplot_titles=[str(result.meta['country']) + " -  " + str(result.meta['name']) for result in results])

    #plot actual and predicted values for each model
    for i in range(0,len(configs)):
        row = 1 + (i // 3)
        col = 1 + (i % 3)

        #get max values for each subplot
        max_y = max(results[i]['Actual'].max(),results[i]['Predicted'].max())
        max_x = results[i].index[-1]
        mean_x = results[i].index.mean()
        
        #get model performance metrics
        rmse = str(round(results[i].meta['rmse'],2))
        norm_rmse = str(round(results[i].meta['norm_rmse'],2))
        r2_score = str(round(results[i].meta['r2_score'],2))
        bic = str(results[i].meta['bic'])

        #plot actual and predicted values
        fig.append_trace(go.Scatter(x=results[i].index, y=results[i]['Actual'], name='Actual',legendgroup=i,marker={'color': 'darkcyan', 'symbol': 104, 'size': 10}), row=row, col=col)
        fig.append_trace(go.Scatter(x=results[i].index, y=results[i]['Predicted'], name='Predicted',legendgroup=i,marker={'color': 'darkorange', 'symbol': 104, 'size': 10}), row=row, col=col)
        
        #add title for y axis annotation with model performance metrics for each subplot and make sure doesn't change plot
        fig.update_yaxes(title_text= results[i].meta['target'], row=row, col=col)


        fig.add_annotation(
                           xref="x"+str(i+1), 
                           yref="y"+str(i+1),
                           yanchor="bottom",
                           borderpad=1,
                           x = mean_x,
                           y = 1 + max_y, 
                           font=dict(size=10),
                           bgcolor="white",
                           bordercolor='black', 
                           borderwidth=1,
                           text="RMSE: " + rmse + " | Normalized RMSE: " + norm_rmse + " | R^2: " + r2_score + " | BIC: " + bic,showarrow=False,align="center")
        
        #add a vertical line to show where the training data ends
        fig.add_shape(type="line",x0=results[i].index[int(0.8*len(results[i].index))],y0=0,x1=results[i].index[int(0.8*len(results[i].index))],y1=1,row=row,col=col,line=dict(color="RoyalBlue",width=1,dash="dashdot",))
    
    #hide legend for all but first plot
    for trace in fig['data']:
        if (int(trace['legendgroup']) > 0):
            trace['showlegend'] = False

    #update layout
    fig.update_xaxes(title_text="Date")
    fig.update_layout(title='Model Evaluation - Predictions vs. Actuals and Metrics',height = rows * 300)
    
    fig.show()

    return results

def r2_compare(results):
    '''
    Given a list of results, this function will compare the r2 scores of each model on a bar chart.
    The function will return a dataframe of r2 scores for each model.

    Parameters
    ----------
    results : list of results
        A list of results for each model.

    Returns
    -------
    r2_df : dataframe
        A dataframe of r2 scores for each model.

    '''

    # create dataframe to store r2 scores
    r2_df = pd.DataFrame(columns=['model_name', 'r2_score'])

    #add row for each model in results
    for result in results:
       
        r2_df = r2_df.append({'model_name': result.meta['name'], 'r2_score': r2_score(result['Actual'], result['Predicted']),'r2_bar':result.meta['r2_bar']}, ignore_index=True)

    #plot r2 scores with plotly
    fig = go.Figure()

    r2_df['color'] = np.where(r2_df['r2_score'] < 0.0, 'indianred', 'darkcyan')


    fig.add_trace(go.Bar(x=r2_df['model_name'], y=r2_df['r2_bar'], name='Normalized RMSE scores',text=round(r2_df['r2_score'],2),textposition='auto',marker_color=r2_df['color']))

    fig.update_layout(barmode='stack', xaxis={'categoryorder':'total descending'})

    fig.update_layout(title='R^2 scores for each model',xaxis_title="Model",yaxis_title="R^2 Score",width = 800)
    
    fig.show()
    
    return r2_df

def error_compare(results):
    '''
    Given a list of results, this function will compare the normalized rmse scores of each model on a bar chart.
    The function will return a dataframe of normalized rmse scores for each model.
    
    Parameters
    ----------
    results : list of results
        A list of results for each model.

    Returns
    -------
    error_df : dataframe
        A dataframe of normalized rmse scores for each model.

    '''
    
    # create dataframe to store error scores
    error_df = pd.DataFrame(columns=['model_name', 'model_nrmse'])

    #add row for each model in results to dataframe
    for result in results:
       
        error_df = error_df.append({'model_name': result.meta['name'], 'model_nrmse': result.meta['norm_rmse']}, ignore_index=True)

    #plot rmse scores with plotly
    fig = go.Figure()

    #set color of bar based on whether it is above or below the model_nrmse of the first model
    error_df['color'] = np.where(error_df['model_nrmse'] > results[0]['Actual'].mean(), 'indianred', 'darkcyan')

    # add bar chart to plot error scores
    fig.add_trace(go.Bar(x=error_df['model_name'], y=error_df['model_nrmse'], name='Normalized RMSE scores',text=round(error_df['model_nrmse'],2),textposition='auto',marker_color=error_df['color']))

    fig.update_layout(barmode='stack', xaxis={'categoryorder':'total descending'})

    fig.update_layout(title='Normalized RMSE scores for each model',xaxis_title="Model",yaxis_title="Normalized RMSE", width = 800)
    
    fig.show()
    
    return error_df
