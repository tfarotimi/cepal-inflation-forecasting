# This file contains the helper functions used in the generation of forecasts. 
# Last update: 08.14.2023 by Inflation Forecasting Farotimi

#add path 
import sys
import os
from tabnanny import verbose
from turtle import mode
sys.path.append('C:\\Users\\Inflation Forecasting\\Documents\\GitHub\\Covid-19-Analysis\\Dev\\Inflation Forecasting\\lib')

from tqdm import tqdm

# EDA Pkgs
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
from datetime import datetime, timedelta

from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.statespace.varmax import VARMAX
from statsmodels.tsa.vector_ar.var_model import VAR

from scipy.stats import boxcox
#implement granger causality test for mex
from statsmodels.tsa.stattools import grangercausalitytests

#pt 
from sklearn.preprocessing import PowerTransformer




import datetime as dt
#plotly
import plotly.graph_objs as go
import plotly.offline as pyo
import plotly.io as pio
from plotly.subplots import make_subplots
pyo.init_notebook_mode()
import openpyxl



#scikit-learn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import mean_squared_error

#auto_arima
import pmdarima as pm
#adf pmdarima
from pmdarima.arima import ADFTest

#adfuller test
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import kpss

#scipy peak detection
from scipy.signal import find_peaks, peak_prominences, peak_widths

import yaml 

#regex
import re

 ###############################################################################################################
 ############################################ HELPER FUNCTIONS #################################################
 ###############################################################################################################                           

def load_data(data_filepath):
    '''
    This function loads the data from the specified filepath and returns a dataframe.

    Parameters
    ----------
    data_filepath : str
        Filepath to the data.

    Returns
    ------- 
    data : dataframe
        Dataframe containing the data.
   '''

    for i in tqdm(range(100)):
        data = pd.read_csv(data_filepath, encoding = 'utf-8',on_bad_lines='skip')

    #rename Unnamed:0 column to Date
    data.rename(columns={'Unnamed: 0':'Date'}, inplace=True)

    #set all time series to monthly frequency
    data.asfreq('MS')

    #convert Date to a datetime object with format %Y%M%DD
    data['Date'] = pd.to_datetime(data['Date'])
    data.set_index('Date', inplace=True)

    return data


#Do EDA

#check for missing values
def check_missing_values(data):
    #visualize missing values
    msno.matrix(data)
    plt.show()
   


#plot indicators for each country
def plot_country_data(data):
    #do line plot for every column in chile_data in a 3X3 grid
    fig, axs = plt.subplots(4, 4, figsize=(15, 15))

    for i, col in enumerate(data.columns):
        if col != 'Date':
            ax = axs[(i - 1) // 4][i % 4]
            ax.plot(data.index, data[col])
            ax.title.set_text(col)
    plt.show()


#plot distribution of indicators for each country
def plot_distribution(data):
    #plot distribution of indicators for each country
    fig, axs = plt.subplots(4, 4, figsize=(15, 15))

    for i, col in enumerate(data.columns):
        if col != 'Date':
            ax = axs[(i - 1) // 4][i % 4]
            sns.distplot(data[col], ax=ax)
            ax.title.set_text(col)
    plt.show()

#plot correlation matrix
def plot_correlation_matrix(data):
    #plot correlation matrix
    plt.figure(figsize=(15, 15))
    sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
    plt.show()

#plot boxplot
def plot_boxplot(data):
    #plot boxplot
    fig, axs = plt.subplots(4, 4, figsize=(15, 15))

    for i, col in enumerate(data.columns):
        if col != 'Date':
            ax = axs[(i - 1) // 4][i % 4]
            sns.boxplot(data[col], ax=ax)
            ax.title.set_text(col)
    plt.show()

#plot scatterplot
def plot_scatterplot(data):
    #plot scatterplot
    fig, axs = plt.subplots(4, 4, figsize=(15, 15))

    for i, col in enumerate(data.columns):
        if col != 'Date':
            ax = axs[(i - 1) // 4][i % 4]
            sns.scatterplot(data[col], ax=ax)
            ax.title.set_text(col)
    plt.show()


#test stationarity
def stationarity_test(data):
    for col in data.columns:
        print(col)
        result = adfuller(data[col])
        print('ADF Statistic: %f' % result[0])
        print('p-value: %f' % result[1])
        if result[1] > 0.05:
            print('Series is not Stationary')
        else:
            print('Series is Stationary')
        print('--------------------------')
    
#make time series data stationary
def transform_data(data):
    data_diff = data.diff().dropna()
    stationarity_test(data_diff)
    return data_diff

#run the model starting from the earliest date data is available and then starting every 50 time steps after 
#and plot the error as we use more recent data
def plot_error(model, data, lags=0):
    rmse = []
    date = []
    for i in range(1, len(data),100):
        rmse.append(model(data.iloc[i:],lags))
        date.append(data.index[i])

    plotly_plot(date, rmse, 'Date', 'RMSE', 'RMSE vs Start Date of Training Data')
    
#plot nice interactive graph using plotly - pass in any additional traces as a list of dictionaries  
def plotly_plot(x,y,xtitle,ytitle,plot_title,additional_plots=[]):
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    fig.add_trace(go.Scatter(x=x, y=y, name=ytitle, marker={'color': 'red', 'symbol': 104, 'size': 10}))

    if additional_plots:
        for plot in additional_plots:
            fig.add_trace(go.Scatter(x=x, y=plot['values'], name=plot['name'], marker={'color': 'blue', 'symbol': 104, 'size': 10}),secondary_y=True) 


    fig.update_layout(title=plot_title, xaxis_title=xtitle, yaxis_title=ytitle, showlegend=True, width=1000, height=600)    
    fig.show()


#plot the actual vs predicted values using plotly
def plotly_predictions(x, y_actual, y_pred):
    plot_title = 'Actual vs Predicted Values for the last 12 months of Headline CPI YOY Increase'

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y_actual, name='Actual', marker={'color': 'red', 'symbol': 104, 'size': 10}))
    fig.add_trace(go.Scatter(x=x, y=y_pred, name='Predicted', marker={'color': 'blue', 'symbol': 104, 'size': 10}))
    fig.update_layout(title=plot_title, xaxis_title='Date', yaxis_title='Predictions', showlegend=True, width=1000, height=600)    
    #add annotation to chart for final RMSE
    fig.add_annotation(xref = 'paper', yref='paper', x=0.9, y = 0.9,text='RMSE: '+ str(mean_squared_error(y_actual, y_pred,squared=False)))

    fig.show()

#prepare training data for use in model
def prepare_data(data, target, predictors=None, horizon = 12):
    horizon = horizon

    #set training data to only include features of interest and target variable


    #fill missing values with 0
    data = data.fillna(0)

    #split data into train and test sets, test set is the last 12 months of data
    train = data.iloc[:-horizon]
    test = data.iloc[-horizon:]
    

    #if data is a dataframe, split train and test sets into X and y
    if isinstance(data, pd.DataFrame):
        #split train and test sets into X and y
        train_X = train.drop(target, axis=1)
        train_y = train[target]
        test_X = test.drop(target, axis=1)
        test_y = test[target]

   

        return train_X, train_y, test_X, test_y
    
    else:

        return None, train, None, test

#prepare training data for use in model
def prepare_forecast_data(data, target, predictors=None, horizon = 12):
    
    data = data.dropna(subset=[target])
    horizon = horizon


    #set training data to only include features of interest and target variable
    if (predictors):
        features = [i for i in predictors]
        features.append(target)
        data = data[features]

    #fill missing values with 0
    
    if isinstance(data, pd.DataFrame):
        X = data.drop(target, axis=1)
        y = data[target]

        #log transform target variable
        #y = np.log(y)

        return X, y
    else:
        return None, data #np.log(data)

#evaluate model and return results in dataframe
def format_results(country, model_name, target, preds, axxx):

    #evaluate model
    if model_name == 'Random Forest w/ Lags':
       
       

        predictions = preds 
        actuals = axxx
        predictions.index = preds.index 
        actuals.index = axxx.index

        predictions.index 

        # 1) RMSE
        rmse = mean_squared_error(actuals, predictions, squared=False)

        # 2) MAPE (watch out for any zeros in y_true!)
        mape = np.mean(np.abs((actuals - predictions) / actuals)) * 100

        # 3) R² and adjusted floor
        r2 = r2_score(actuals, predictions)
        r2_bar = r2 if r2 >= 0 else -0.1

        print(f"RMSE: {rmse:.4f}")
        print(f"MAPE: {mape:.2f}%")
        print(f"R²: {r2:.3f}, R²_bar: {r2_bar:.3f}")
    else:
        rmse = mean_squared_error(actuals, predictions,squared=False)
        mape = np.mean(np.abs((actuals - predictions) / actuals)) * 100
        r2 = r2_score(actuals, predictions)
        r2_bar = r2 if r2 >= 0 else -0.1
        



    results = pd.concat([actuals,predictions],axis=1)

    results.columns = ['Actual','Predicted'] # if model_name != 'Random Forest w/ Lags' else list("Actual_" + actuals.columns) + list("Predicted_" + predictions.columns)

    #combine ts.columns and preds.columns into new columns




    results.index.name = 'Date'

    residuals = results['Actual'] - results['Predicted'] if model_name != 'Random Forest w/ Lags' else predictions - actuals

    results.meta = {'name': model_name, 'country':country, 'target':target, 'rmse':rmse,'r2_score':r2, 'r2_bar':r2_bar, 'residuals':residuals, 'mape':mape} 

    return results

#create time lags for time series data
def time_delay_embedding(series: pd.Series,
                         n_lags: int,
                         horizon: int,
                         return_Xy: bool = False):
    """
    Time delay embedding
    Time series for supervised learning
    :param series: time series as pd.Series
    :param n_lags: number of past values to used as explanatory variables
    :param horizon: how many values to forecast
    :param return_Xy: whether to return the lags split from future observations
    :return: pd.DataFrame with reconstructed time series
    """
    assert isinstance(series, pd.Series)

    name = series.name if series.name == 'YoY Increase Inflation' else 'YoY Increase Inflation'

    n_lags = n_lags if n_lags else 1

    n_lags_iter = list(range(0, n_lags, 1))

    #n_lags_iter = list(range(5, -5, -1))

    df_list = [series.shift(i) for i in n_lags_iter]
    df = pd.concat(df_list, axis=1)
    df.columns = [f'{name}(t-{j})'
                if j >= 0 else f'{name}(t+{np.abs(j) + 1})'
                for j in n_lags_iter]
    
 


    df.columns = [re.sub('t\-0', 't', x) for x in df.columns]


    if not return_Xy:
        return df

    is_future = df.columns.str.contains('\+')

    X = df.iloc[:, ~is_future]
    Y = df.iloc[:, is_future]
    if Y.shape[1] == 1:
        Y = Y.iloc[:, 0]


    return X, Y

#forecast the next time step 
def step_forecast(_model, history_X, history_y ,next_X,counter,n_steps,params):
    #fit model and make a one-step prediction
    if (_model == "lagged_rf"):
        model = RandomForestRegressor(n_estimators=1000)
        model_fit = model.fit(history_X, history_y)
        
        preds = model.predict(next_X)

        next_pred = preds[0].tolist()

        #get feature importance
        #importance_scores = pd.Series(dict(zip(X_tr.columns, model.feature_importances_)))
        # getting top 20 features
        #top_20_features = importance_scores.sort_values(ascending=False)[:20]
        
    elif (_model == "multivar_ARDL"):
        model = AutoReg(history_y, lags=12, exog=history_X)
        model_fit = model.fit()
        preds = model_fit.predict(dynamic=False, exog=history_X,exog_oos=next_X,start=len(history_y), end=len(history_y))

        next_pred = preds.iloc[-1]

    elif (_model == "univar_ARDL"):
        model = AutoReg(history_y, lags=12)
        model_fit = model.fit()

        preds = model_fit.predict(dynamic=False, start=len(history_y), end=len(history_y))

        next_pred = preds.iloc[-1]

    elif(_model == "arima"):
        model = ARIMA(history_y, order=params['order'],
                        seasonal_order=params['seasonal_order'],
                        enforce_stationarity=False,
                        enforce_invertibility=False)

        model_fit = model.fit()

        preds = model_fit.predict(start = len(history_y), end = len(history_y))

        next_pred = preds.iloc[-1]

    elif(_model == "sarimax"):
        model = SARIMAX(history_y, history_X, order=(1,1,1),
                        seasonal_order=(0,0,2,12),
                        enforce_stationarity=False,
                        enforce_invertibility=False)
       
        model_fit = model.fit()

        preds = model_fit.predict(start = len(history_y), end = len(history_y),exog = next_X)

        next_pred = preds.iloc[-1]

    elif (_model == "varmax"):
        varmax_Y = pd.concat([history_y, history_X], axis=1)

        p, d, q = params['order']
        P, D, Q, M = params['seasonal_order']
    
        if d == 0:
            model = VARMAX(varmax_Y,order=(p,q), seasonal_order=(P,D,Q,12))
        else:
            model = VARMAX(varmax_Y,order=(p,q), seasonal_order=(P,D,Q,12), trend='n')

        model_fit = model.fit(maxiter=100)
        preds = model_fit.get_forecast(steps=1)
        next_pred = preds.predicted_mean.iloc[-1][0]

    elif (_model == "var"):
        p = params['order']
        diff = params['diff']

        var_Y = pd.concat([history_y, history_X], axis=1)

        if diff == 0:
            model = VAR(var_Y)
            model_fit = model.fit(p)
            preds = model_fit.forecast(var_Y.values, steps=1)
            next_pred = preds[0][0]

        elif diff == 1:
            differenced = var_Y.diff().dropna()
            model = VAR(differenced)
            model_fit = model.fit(p)
            next_pred = pd.DataFrame(model_fit.forecast(differenced.values, steps=16), index = next_X.index,columns=var_Y.columns + '_1d')
        

        elif diff == 2:
            differenced = var_Y.diff().diff().dropna()
            model = VAR(differenced)
            model_fit = model.fit(p)
            next_pred = pd.DataFrame(model_fit.forecast(differenced.values, steps=16), index = next_X.index,columns=var_Y.columns + '_2d')
        elif diff == 3:
            differenced = var_Y.diff().diff().diff().dropna()
            model = VAR(differenced)
            model_fit = model.fit(p)
            next_pred = pd.DataFrame(model_fit.forecast(differenced.values, steps=16), index = next_X.index,columns=var_Y.columns + '_3d')
        elif diff == 4:
            differenced = var_Y.diff().diff().diff().diff().dropna()
            model = VAR(differenced)
            model_fit = model.fit(p)
            next_pred = pd.DataFrame(model_fit.forecast(differenced.values, steps=16), index = next_X.index,columns=var_Y.columns + '_4d')

    else: 
        print("Model not found")

    
        
    if counter == n_steps - 1:

        return model_fit, next_pred
    else:
        return None, next_pred
    
def prepare_rf_lagged_only(data, lags=12, horizon=12):
    """
    Prepares lagged features using only the target inflation variable (e.g., YoY inflation)
    for training a Random Forest model.

    Parameters:
        data (pd.Series or pd.DataFrame): Time series of target variable (e.g., inflation)
        lags (int): Number of lagged observations to use as features
        horizon (int): Forecast horizon (number of months ahead to predict)

    Returns:
        X (pd.DataFrame): Feature matrix with lagged variables
        y (pd.Series): Target variable (future inflation), named 'YoY Increase Inflation'
    """
    # Ensure input is a Series and name it correctly
    if isinstance(data, pd.DataFrame):
        series = data.iloc[:, 0]
    else:
        series = data

    series = series.rename("YoY Increase Inflation")
    series = series.dropna().reset_index(drop=True)

    # Create lagged DataFrame
    df = pd.DataFrame()
    df['target'] = series.shift(-horizon)
    for i in range(1, lags + 1):
        df[f'lag_{i}'] = series.shift(i)

    df = df.dropna().reset_index(drop=True)

    X = df[[f'lag_{i}' for i in range(1, lags + 1)]]
    y = df['target']
    y.name = "YoY Increase Inflation"  # <- Ensure output y has correct name

    #.index    
    # Ensure X and y are aligned
    X.index = data.index[-len(X):]  # Align X with the original index of the data
    y.index = data.index[-len(y):]  # Align y with the last n rows of X


    return X, y



def remove_pulses(data):
    #find peaks
    smooth_data = data.copy()
    data = data.squeeze()
    
    #set index to integers
    data.index = range(len(data))
    peaks, _ = find_peaks(data, height= data.mean() + 2 * data.std())

    prominences = peak_prominences(data, peaks)[0]
    pulse_widths = peak_widths(data, peaks, rel_height=0.9)

    for i in range(len(pulse_widths[0])):
        left_bound = pulse_widths[2][i].astype(int)
        right_bound = pulse_widths[3][i].astype(int)

        smooth_data.iloc[left_bound:right_bound] = np.nan

    fig, ax = plt.subplots(nrows=1, ncols=2)
    

    ax[0].plot(data)
    ax[0].plot(peaks, data[peaks], "x")

    #plot prominences
    ax[0].vlines(x=peaks, ymin=data[peaks] - prominences, ymax = data[peaks], color = "C1")

    #plot peak widths
    ax[0].hlines(y=pulse_widths[1], xmin=pulse_widths[2], xmax=pulse_widths[3], color = "C1")

    ax[0].set_title("Original Data - Outliers to be removed")


    

    smooth_data = smooth_data.interpolate(method='linear', limit_direction='forward', axis=0)

    ax[1].plot(smooth_data)
    ax[1].set_title("Smoothed Data")

    fig.tight_layout()
    plt.show()

    return smooth_data


def auto_grid_search(model, data):

    from statsmodels.tsa.stattools import kpss
    import pmdarima as pm
    
    data = data.dropna()

    # Training data: everything except the last 12 months
    train_data = data.iloc[:-12]

    # You already provide order and seasonal_order explicitly; no need for auto_arima to search further
    if model == 'arima':
        print(f"Trying ARIMA grid search...")

        auto = pm.auto_arima(data, seasonal=True, m=12,
                                start_p=0, start_q=0,
                                max_p=5, max_q=5, # Increased max orders
                                start_P=0, start_Q=0,
                                max_P=3, max_Q=3, # Increased max seasonal orders
                                d=None, D=None, # Let auto_arima determine differencing orders
                                max_d=2, max_D=2, # Max differencing orders to consider
                                trace=False, # Set to True to see the search process
                                error_action='ignore', # Ignore warnings/errors for individual models
                                suppress_warnings=True, # Suppress convergence warnings
                                stepwise=True, # Use stepwise algorithm for faster search
                                scoring='mse',
                                random_state=42 # Use AIC for model selection
                                ) # Optimize for RMSE
        params = [auto.get_params()['order'], auto.get_params()['seasonal_order']]
        return params

        
    elif model == 'sarimax':
        print("searching for best parameters for SARIMAX model...")
       
        auto = pm.auto_arima(data, seasonal=True, m=12,
                                start_p=0, start_q=0,
                                max_p=5, max_q=5, # Increased max orders
                                start_P=0, start_Q=0,
                                max_P=3, max_Q=3, # Increased max seasonal orders
                                d=None, D=None, # Let auto_arima determine differencing orders
                                max_d=2, max_D=2, # Max differencing orders to consider
                                trace=False, # Set to True to see the search process
                                error_action='ignore', # Ignore warnings/errors for individual models
                                suppress_warnings=True, # Suppress convergence warnings
                                stepwise=True, # Use stepwise algorithm for faster search
                                scoring='mse',
                                random_state=42 # Use AIC for model selection
                                ) # Optimize for RMSE
        
    #make predictions
    params = [auto.get_params()['order'], auto.get_params()['seasonal_order']]
    return params

def arima_dateSearch(data, country, target, params):
    last_train_date = data.index[-12] - pd.DateOffset(months=12)
    begin_date = data.index[0]
    target = data.columns[-1]

    rmse_list = pd.DataFrame()

    print("Running ARIMA model to find best date to start training...")
    for i in tqdm(pd.date_range(begin_date, last_train_date, freq='MS')):
        

        try:
            rmse = ARIMA(data[i:], country, target, params).meta['rmse']
            

            #add rmse to list
            rmse_list = rmse_list.append(pd.DataFrame([rmse], columns=['rmse'], index=[i]))
            #get index of min rmse
        
            min_rmse_index = rmse_list['rmse'].idxmin().strftime('%Y-%m-%d')
        except:
            print( i, last_train_date)
            print ("error - returning index with lowest rmse")
            print ("failed on: ", i, last_train_date)
            print ("rmse_list: ", rmse_list)
            break 
            
          
    print (rmse_list)
    return min_rmse_index    

def write_params_to_file(country, start_dt, end_dt, params):
    #read yaml file
    with open(r'C:\Users\tfarotimi\Documents\CEPAL\Inflation Forecasting\dat\forecast_params.yaml', 'r') as file:
        obj = yaml.load(file, Loader=yaml.FullLoader)

    #add new params to yaml file
    obj[country] = {'start_date':start_dt.strftime('%Y-%m-%d'), 'end_date': end_dt.strftime('%Y-%m-%d'), 'params': params}
    

    #write to yaml file

    with open(r'C:\Users\tfarotimi\Documents\CEPAL\Inflation Forecasting\dat\forecast_params.yaml', 'w') as file:
        parameters = yaml.dump(obj, file, sort_keys=False)


def boxcox_transform(X):

    const = np.abs(min(X)) + 0.000001 #add a small constant to make all values positive
    X = X + const

    transformed, lam = boxcox(X)
    print('çonst'': %f' % const)
    print('Lambda: %f' % lam)


    return transformed, lam, const

def boxcox_inverse(df, lam, const):
 
    if lam == 0:
        final =  np.exp(df) - const
        return final 
    
    final = np.exp(np.log(lam * df + 1) / lam) - const

    return final

def overfitting_test(model, data, params):
    """
    This function tests for overfitting by comparing the R2 of the model on the training data and the R2 of the model on the test data
    If the R2 of the model on the training data is much higher than the R2 of the model on the test data, then the model is overfitting
    """
    #split data dataframe into five batches
    #split data into rolling window batches of length 60'
    batches = [data[i:i+60] for i in range(0, len(data), 12)]
    print(len(batches))
    #split data into training and test data
    diff_list = []
    for data in batches:
        train = data[:int(0.8*(len(data)))]
        test = data[int(0.8*(len(data))):]
        print(train.shape, test.shape)
        #fit model on training data
        model = ARIMA(train, order = params[0], seasonal_order = params[1])
        model_fit = model.fit()
        
        #get predictions for training and test data
        train_pred = model_fit.predict()
        test_pred = model_fit.predict(start = len(train), end = len(data)-1)
        
        #get mse for training and test data
        train_rmse = np.sqrt(mean_squared_error(train, train_pred))
        test_rmse = np.sqrt(mean_squared_error(test, test_pred))
    
        
        #return the difference between the R2 of the model on the training data and the R2 of the model on the test data
        print(train_rmse, test_rmse)
        diff =  train_rmse - test_rmse
        diff_list.append(diff)

    #plot the difference between the R2 of the model on the training data and the R2 of the model on the test data
    
    print(diff_list)
    plt.plot(diff_list)

'''
def create_forecast_chart_doc(model, forecast_chart_list):
    timestamp = dt.datetime.now().strftime("%Y%m%d%H%M%S")
    #hour 

    # Your existing code here
    
    # Create a new Word document
    document = Document()

    # Loop through each chart in forecast_chart_list
    for i, chart in enumerate(forecast_chart_list):
       #add a picture to the document from a file path
        document.add_picture(chart, width=Inches(6.0))       

    # Save the Word document
    document.save(model + "_forecast_report_" + timestamp + ".docx")

'''

def read_inflation_data(FILE_NAME, EXOG_FILE_NAME=None):

    df = pd.read_excel(FILE_NAME, sheet_name='IPCg_m_via', header=5, nrows=57)

    df= df.dropna(axis=1, how='all')

    df.set_index('CCC', inplace=True)

    SUR_data = {}
    CEN_data = {}
    CAR_data = {}
    REG_data = {}

    for row in df.index: 
        if df.loc[row, 'Cod3reg'] == 'SUR':
            SUR_data[row] = pd.DataFrame(columns=["Period", "YoY Increase Inflation"])
            
            for col in df.columns[14:]:
                SUR_data[row].loc[len(SUR_data[row])] = [col, df.loc[row, col]]

        elif df.loc[row, 'Cod3reg'] == 'CEN':
            CEN_data[row] = pd.DataFrame(columns=["Period", "YoY Increase Inflation"])

            for col in df.columns[14:]:
                CEN_data[row].loc[len(CEN_data[row])] = [col, df.loc[row, col]]

        elif df.loc[row, 'Cod3reg'] == 'CAR':
            CAR_data[row] = pd.DataFrame(columns=["Period", "YoY Increase Inflation"])

            for col in df.columns[14:]:
                CAR_data[row].loc[len(CAR_data[row])] = [col, df.loc[row, col]]
        else:
            if isinstance(df.loc[row,'Cod3reg'], str):
                REG_data[row] = pd.DataFrame(columns=["Period", "YoY Increase Inflation"])

                for col in df.columns[14:]:
                    REG_data[row].loc[len(REG_data[row])] = [col, df.loc[row, col]]

    all_data = [SUR_data, CEN_data, CAR_data, REG_data]
        
    for region in all_data: 
        for key in region.keys():
            region[key]['Month'] = region[key]['Period'].apply(lambda x: x.split('_')[1][1:])
            region[key]['Year'] = region[key]['Period'].apply(lambda x: x.split('_')[0])

            region[key]['Date'] = region[key]['Year'] + '-' + region[key]['Month'] + '-01'

            region[key]['Date'] = pd.to_datetime(region[key]['Date'])
            
            
            region[key].set_index('Date', inplace=True)

            #rename index to Period
            region[key].index.name = 'Period'

            #drop Year and Month columns
            region[key].drop(['Year', 'Month','Period'], axis=1, inplace=True)

            region[key] = region[key].dropna()

            # if region[key]['YoY Increase Inflation'].loc[:'1996'].max() > 200:
            #     region[key] = region[key].loc['1996':]

            # if region[key]['YoY Increase Inflation'].loc[:'2000'].max()>200:
            #     region[key] = region[key].loc['2000':]

            # if region[key]['YoY Increase Inflation'].loc['2005':].max()>200:
            #     region[key] = region[key].loc['2005':]

    if EXOG_FILE_NAME is not None:
        for region in all_data: 
            for key in region.keys():
                region[key] = merge_exogenous_data(key, region[key], EXOG_FILE_NAME)

        
    return SUR_data, CEN_data, CAR_data, REG_data



            
        
def merge_exogenous_data(country, target_df, EXOG_FILE_NAME):
    
    wb = openpyxl.load_workbook(EXOG_FILE_NAME)

    country_data = {}

    for sheet in wb.sheetnames:
        if country == sheet:
            country_data[sheet] = pd.read_excel(EXOG_FILE_NAME, sheet_name  = sheet)
            country_data[sheet].set_index('Period', inplace = True)   

            #GET PCT CHANGE OF 2ND COLUMN AND REPLACE 2ND COLUMN WITH IT
            for i in range(1,len(country_data[sheet].columns)):

                #drop empty columns
                
            
                #linearly interpolate missing values in each column forwards 
                country_data[sheet][country_data[sheet].columns[i]] = country_data[sheet][country_data[sheet].columns[i]].interpolate(method='linear', limit_direction='forward', axis=0)

                

                country_data[sheet][country_data[sheet].columns[i]] = country_data[sheet][country_data[sheet].columns[i]].pct_change(12) * 100

                #replace inf values with NA
                country_data[sheet][country_data[sheet].columns[i]] = country_data[sheet][country_data[sheet].columns[i]].replace([np.inf, -np.inf], np.nan)

                #fill missing values with linearly interpolated values

            #drop empty columns
            country_data[sheet] = country_data[sheet].dropna(axis = 1, how = 'all')

            #drop Yoy increase inflation column
            country_data[sheet] = country_data[sheet].drop(country_data[sheet].columns[0], axis = 1)
            
            #drop NA values
            #rplace inf values with NA
            target_df = target_df.merge(country_data[sheet], how = 'left', left_index = True, right_on = 'Period')

            target_df = target_df.set_index('Period')

        else:
            target_df = target_df

    return target_df
            
def grangers_causation_matrix(data, variables, test='ssr_chi2test', verbose=False):
    """Check Granger Causality of all possible combinations of the Time series.
    The rows are the response variable, columns are predictors. The values in the table 
    are the P-Values. P-Values lesser than the significance level (0.05), implies 
    the Null Hypothesis that the coefficients of the corresponding past values is 
    zero, that is, the X does not cause Y can be rejected.

    data      : pandas dataframe containing the time series variables
    variables : list containing names of the time series variables.
    """

    maxlag=12
    test = 'ssr_chi2test'

    df = pd.DataFrame(np.zeros((len(variables), len(variables))), columns=variables, index=variables)
    for c in df.columns:
        for r in df.index:
            #ensure stationarity of data
            stationary_results = [adfuller_test(data[r]), adfuller_test(data[c])]
            if all(stationary_results) < 0.05:
                print("testing for granger causality with stationary series")
                final_data = data[[r,c]]
                test_result = grangercausalitytests(final_data, maxlag=maxlag, verbose=False)
                p_values = [round(test_result[i+1][0][test][1],4) for i in range(maxlag)]
            else:
                data_2_r = data[[r]].diff().dropna()
                data_2_c = data[[c]].diff().dropna()
                stationary_results = [adfuller_test(data_2_r), adfuller_test(data_2_c)]
                if all(stationary_results) < 0.05:
                    print("differenced 1x - testing for granger causality with stationary series")
                    final_data = pd.concat([data_2_r, data_2_c], axis=1).dropna()
                    test_result = grangercausalitytests(final_data, maxlag=maxlag, verbose=False)
                    p_values = [round(test_result[i+1][0][test][1],4) for i in range(maxlag)]
                else:
                    data_3_r = data_2_r.diff().dropna()
                    data_3_c = data_2_c.diff().dropna()
                    stationary_results = [adfuller_test(data_3_r), adfuller_test(data_3_c)]
                    if all(stationary_results) < 0.05:
                        print("differenced 2x - testing for granger causality with stationary series")
                        final_data = pd.concat([data_3_r, data_3_c], axis=1).dropna()
                        test_result = grangercausalitytests(final_data, maxlag=maxlag, verbose=False)
                        p_values = [round(test_result[i+1][0][test][1],4) for i in range(maxlag)]
                    else:
                        print("data is not stationary", r, c)
                        p_values = [1] * maxlag
                        break
            
            if verbose: print(f'Y = {r}, X = {c}, P Values = {p_values}')
            min_p_value = np.min(p_values)
            df.loc[r, c] = min_p_value
    df.columns = [var + '_x' for var in variables]
    df.index = [var + '_y' for var in variables]

    #get all 
  



    #make heatmap and show it where values > 0.1
   
    return df
        
def get_best_predictors(matrix):
    """ 
    This function takes in a granger causality matrix and returns the best predictors for the target variable
    """

    best_predictors = []
    for col in matrix.index:
        #remove '_x' from column name
        col = col.replace('_x', '')
        best_predictors.append(col)
    return best_predictors

def create_predictor_combinations(predictors):
    #add 'YoY Increase Inflation' to the list of predictors in first position
    predictors.insert(0, 'YoY Increase Inflation')
    from itertools import combinations
    combs_array = []
    for i in range(1,len(predictors)+1):
        combs = list(combinations(predictors,i))
        combs = [comb for comb in combs if comb[0] == 'YoY Increase Inflation' and len(comb) > 1]
        if len(combs) > 0:  
            combs_array.append(combs) 

    combs_array = [comb for sublist in combs_array for comb in sublist]
    
    return combs_array

#run cointegration test and remove non-cointegrated variables
from statsmodels.tsa.vector_ar.vecm import coint_johansen

def cointegration_test(df, alpha=0.05):
    """Perform Johanson's Cointegration Test and Report Summary"""
    out = coint_johansen(df,-1,5)
    d = {'0.90':0, '0.95':1, '0.99':2}
    traces = out.lr1
    cvts = out.cvt[:, d[str(1-alpha)]]
    def adjust(val, length= 6): return str(val).ljust(length)

    # Summary
    print('Name   ::  Test Stat > C(95%)    =>   Signif  \n', '--'*20)
    for col, trace, cvt in zip(df.columns, traces, cvts):
        print(adjust(col), ':: ', adjust(round(trace,2), 9), ">", adjust(cvt, 8), ' =>  ' , trace > cvt)

#test mex for stationarity 
from statsmodels.tsa.stattools import adfuller

def adfuller_test(series, signif=0.05, name='', verbose=False):
    """Perform ADFuller to test for Stationarity of given series and print report"""
    r = adfuller(series, autolag='AIC')
    output = {'test_statistic':round(r[0], 4), 'pvalue':round(r[1], 4), 'n_lags':round(r[2], 4), 'n_obs':r[3]}
    p_value = output['pvalue']
    def adjust(val, length= 6): return str(val).ljust(length)

 
    
    return p_value

def invert_transformation(df_train, df_forecast, diff = 0):
    """Revert back the differencing to get the forecast to original scale."""
    df_fc = df_forecast.copy()
    columns = df_train.columns
    for col in columns:        
        # Roll back 2nd Diff
        if diff == 4:
            df_fc[str(col)+'_3d'] = (df_train[col].iloc[-3]-df_train[col].iloc[-4]) + df_fc[str(col)+'_4d'].cumsum()
            df_fc[str(col)+'_2d'] = (df_train[col].iloc[-2]-df_train[col].iloc[-3]) + df_fc[str(col)+'_3d'].cumsum()
            df_fc[str(col)+'_1d'] = (df_train[col].iloc[-1]-df_train[col].iloc[-2]) + df_fc[str(col)+'_2d'].cumsum()
   
        if diff == 3:
            df_fc[str(col)+'_2d'] = (df_train[col].iloc[-2]-df_train[col].iloc[-3]) + df_fc[str(col)+'_3d'].cumsum()
            df_fc[str(col)+'_1d'] = (df_train[col].iloc[-1]-df_train[col].iloc[-2]) + df_fc[str(col)+'_2d'].cumsum()
        if diff == 2:
            df_fc[str(col)+'_1d'] = (df_train[col].iloc[-1]-df_train[col].iloc[-2]) + df_fc[str(col)+'_2d'].cumsum()
        # Roll back 1st Diff
        df_fc[str(col)+'_forecast'] = df_train[col].iloc[-1] + df_fc[str(col)+'_1d'].cumsum()
    if diff == 1:
        df_fc = df_fc[df_fc.columns.drop(list(df_fc.filter(regex='_1d')))]
    if diff == 2:
        df_fc = df_fc[df_fc.columns.drop(list(df_fc.filter(regex='_2d')))]
        df_fc = df_fc[df_fc.columns.drop(list(df_fc.filter(regex='_1d')))]
    if diff == 3:
        df_fc = df_fc[df_fc.columns.drop(list(df_fc.filter(regex='_3d')))]
        df_fc = df_fc[df_fc.columns.drop(list(df_fc.filter(regex='_2d')))]
        df_fc = df_fc[df_fc.columns.drop(list(df_fc.filter(regex='_1d')))]
    if diff == 4:
        df_fc = df_fc[df_fc.columns.drop(list(df_fc.filter(regex='_4d')))]
        df_fc = df_fc[df_fc.columns.drop(list(df_fc.filter(regex='_3d')))]
        df_fc = df_fc[df_fc.columns.drop(list(df_fc.filter(regex='_2d')))]
        df_fc = df_fc[df_fc.columns.drop(list(df_fc.filter(regex='_1d')))]
    return df_fc

def forecast_accuracy(forecast, actual):
    mape = np.mean(np.abs(forecast - actual)/np.abs(actual))  # MAPE
    rmse = np.mean((forecast - actual)**2)**.5  # RMSE
    corr = np.corrcoef(forecast, actual)[0,1]   # corr
    return({'mape':mape,'rmse':rmse, 'corr':corr})


from statsmodels.stats.stattools import durbin_watson


def run_var_for_all_combos(df, predictors):

    results_df = pd.DataFrame(columns = ['Combination', 'RMSE', 'MAPE', 'Correlation', 'Stationary', 'Times Differenced', 'Error Correlation'])

    combination_list = create_predictor_combinations(predictors)

    print(len(combination_list))

    lowest_mape = 1000000

    for comb in tqdm(combination_list):
       
        comb_df = df[list(comb)]

        train = comb_df[:-12]
        test = comb_df[-12:]

        # ADF Test on 'YoY Increase Inflation'
        all_p_values = []
        diff = 0

        for col in train.columns:
            p_value = adfuller_test(train[col])
            all_p_values.append(p_value)

        #if any p-value is greater than 0.05, difference the series
        if np.any(np.array(all_p_values) > 0.05):
            differenced = train.diff().dropna()
           
            diff = 1
            all_p_values = []
                
            for col in differenced.columns:
                p_value = adfuller_test(differenced[col])
                all_p_values.append(p_value)

            if np.any(np.array(all_p_values) > 0.05):
                stationary = False
                differenced = differenced.diff().dropna()
                diff = 2

                all_p_values = []
                for col in differenced.columns:
                    p_value = adfuller_test(differenced[col])
                    
                    all_p_values.append(p_value)
            
                if np.any(np.array(all_p_values) > 0.05):
                    stationary = False
                else:
                    stationary = True

            else:
                stationary = True
                
        else:
            stationary = True



        #determine best lag order for VAR model
        if diff == 0:
            model = VAR(train)
        else:
            model = VAR(differenced)
        try:
            x = model.select_order(maxlags=12)
            #give best lag order
        except:
            try:
                x = model.select_order(maxlags=10)
            except:
                try:
                    x = model.select_order(maxlags=8)
                except:
                    try:
                        x = model.select_order(maxlags=6)
                    except:
                        try:
                            x = model.select_order(maxlags=4)
                        except:
                            try:
                                x = model.select_order(maxlags=2)
                            except:
                                try:
                                    x = model.select_order(maxlags=1)
                                except:
                                    print("Error on combination: ", comb)
                                    eval = {'mape':np.nan,'rmse':np.nan, 'corr':np.nan}
                                continue;
                        
        lag_order = x.selected_orders['aic']
        lag_order

        #fit the VAR model
        model_fitted = model.fit(lag_order)

        #check for serial correlation of residuals using Durbin Watson Statistic
        out = durbin_watson(model_fitted.resid)

        # for col, val in zip(differenced.columns, out):
        #     print((col), ':', round(val, 2))

        #make predictions on validation
        try:
            prediction = model_fitted.forecast(differenced.values, steps=16)
        except:
            #go to next combination if forecast fails
            print("Error on combination: ", comb)
            eval = {'mape':np.nan,'rmse':np.nan, 'corr':np.nan}
            prediction = np.nan
            continue;
        
        prediction = pd.DataFrame(prediction, index = test.index, columns = test.columns + '_pred')

        # Get the lag order
        lag_order = model_fitted.k_ar

        # Input data for forecasting
        forecast_input = differenced.values[-lag_order:]

        fc = model_fitted.forecast(y=forecast_input, steps=16)
        
        if (diff == 1):
            df_forecast = pd.DataFrame(fc, index=test.index, columns=test.columns + '_1d')
            df_results = invert_transformation(train, df_forecast, diff = diff)   
        elif (diff == 2):
            df_forecast = pd.DataFrame(fc, index=test.index, columns=test.columns + '_2d')
            df_results = invert_transformation(train, df_forecast, diff = diff)
        elif (diff == 3):
            df_forecast = pd.DataFrame(fc, index=test.index, columns=test.columns + '_3d')
            df_results = invert_transformation(train, df_forecast, diff = diff)
        elif (diff == 4):
            df_forecast = pd.DataFrame(fc, index=test.index, columns=test.columns + '_4d')
            df_results = invert_transformation(train, df_forecast, diff = diff)
    
        
             

        eval = forecast_accuracy(df_results['YoY Increase Inflation_forecast'].values, df['YoY Increase Inflation'][-12:].values)

        #calculate durbin watson statistic on residuals, interpret and add to results_df
        if out[0] < 1.5 or out[0] > 2.5:
            eval['dw'] = 'Serial Correlation'
        else:
            eval['dw'] = 'No Serial Correlation'
        
       
        results_df = results_df.append({'Combination': comb, 'RMSE': eval['rmse'], 'MAPE': eval['mape'], 'Stationary': stationary, 'Times Differenced': diff, 'Correlation': eval['corr'], 'Error Correlation': eval['dw']}, ignore_index=True)

        if eval['mape'] < lowest_mape:
            lowest_mape = eval['mape']
            best_comb = comb

        print("Lowest MAPE: ", lowest_mape, "Best Combination: ", best_comb)


    #sort results by RMSE lowest to highest and return
    results_df = results_df.sort_values(by = 'RMSE')
    pd.set_option('max_colwidth', 800)
    
    return results_df

#make sure forecast is never less than 2% or g


#impulse response function gird 
def impulse_response_grid(irf_list):
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 12))
    for i, (irf, ax) in enumerate(zip(irf_list, axes.flatten())):
        irf.plot(ax=ax)
        ax.set_title(f'Impulse Response Functions to {irf.orth_irf.ort_irf[0]}')
        ax.set_xlabel('Lag')
        ax.set_ylabel('Response')
        ax.set_xticks(range(1, 13))

def extend_monthly_series_to_match_annual_avg(actual_months, annual_target_2025, annual_target_2026, total_months=12):
    """
    Extend a partial monthly time series to match a known annual average.

    Parameters:
    - actual_months: pd.Series with datetime index and monthly actual values
    - annual_target: float, target average for the entire year (e.g. IMF projection)
    - total_months: int, usually 12

    Returns:
    - pd.Series of length 12 with filled values
    """

    # Step 1: Number of actual and missing months
    n_actual = len(actual_months)
    n_missing = total_months - n_actual

    n_missing_2026 = 12 - n_actual

    if n_actual >= total_months:
        return actual_months.iloc[:total_months]  # no extension needed

    # Step 2: Total value needed for the year
    total_required = annual_target_2025 * total_months
    total_actual = actual_months.sum()
    total_needed = total_required - total_actual

    total_required_2026 = annual_target_2026 * total_months
    total_needed_2026 = total_required

    # Step 3: Equal distribution for missing months
    projected_value = total_needed / n_missing
    last_month = actual_months.index[-1]
    forecast_index = pd.date_range(start=last_month + pd.DateOffset(months=1), periods=n_missing, freq='MS')

    future_months = pd.Series([projected_value] * n_missing, index=forecast_index)
    full_series = pd.concat([actual_months, future_months])

    return full_series

def interpolate_monthly_from_annual(actual_months, annual_forecasts, add_noise=False, seasonal_amp=0.0):
    """
    Generate monthly series that match annual averages, optionally with noise or seasonality.

    Parameters:
    - actual_months: pd.Series with datetime index
    - annual_forecasts: dict like {2025: 85.0, 2026: 87.5}
    - add_noise: bool, add Gaussian noise
    - seasonal_amp: float, amplitude for sinusoidal seasonality

    Returns:
    - full_monthly_series: pd.Series
    """
    result = actual_months.copy()
    last_actual = actual_months.index[-1] if not actual_months.empty else None

    for year, target in annual_forecasts.items():
        year_index = pd.date_range(start=f"{year}-01-01", end=f"{year}-12-01", freq="MS")
        
        if last_actual and last_actual.year == year:
            known = result[result.index.year == year]
            n_known = len(known)
            n_missing = 12 - n_known
            total_known = known.sum()
        else:
            known = pd.Series(dtype=float)
            n_known = 0
            n_missing = 12
            total_known = 0

        if n_missing > 0:
            avg_fill = (target * 12 - total_known) / n_missing
            base_values = np.full(n_missing, avg_fill)

            # Optionally add sinusoidal seasonality
            if seasonal_amp > 0:
                months = np.arange(n_known + 1, 13)
                seasonality = seasonal_amp * np.sin(2 * np.pi * months / 12)
                base_values += seasonality

            # Optionally add noise
            if add_noise:
                noise = np.random.normal(0, 0.5, n_missing)  # tune stddev as needed
                base_values += noise

            fill_index = year_index[n_known:]
            filled = pd.Series(base_values, index=fill_index)
            result = pd.concat([result, filled])

    return result.sort_index()


def add_lags(series, n_lags=3):
    """
    Add n lagged versions of a series as columns.

    Parameters:
    - series: pd.Series
    - n_lags: int
    - var_name: str, base name for the variable

    Returns:
    - pd.DataFrame with lagged columns
    """
 
    var_name = series.name
    #get letters until '(' is found 
    if '(' in var_name:
        var_name = var_name.split('(t-0')[0]
    else: 
        raise ValueError("series name must be provided and contain \'(t-0)\' to identify the variable name")

    df = pd.DataFrame({f"{var_name}(t-{i})": series.shift(i) for i in range(1, n_lags + 1)})
    df[var_name + "(t-0)"] = series
    return df.dropna()