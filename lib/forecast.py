from unicodedata import name
from matplotlib.pylab import rand
import numpy as np

import os
import datetime as dt
import seaborn as sns
import pmdarima as pmd
#EXP SMOOTHING
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sympy import Max
import yaml

#add path to sys so that we can import functions from other folders
import sys
sys.path.append(r"C:\Users\tfarotimi\Documents\CEPAL\Inflation Forecasting\src")
sys.path.append(r"C:\Users\tfarotimi\Documents\CEPAL\Inflation Forecasting\lib")

#import libraries
import pandas as pd

#graphing libraries
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objs as go
import plotly.io as pio
import pdb

#Python Image Library
from PIL import Image, ImageOps

#pathlib
from pathlib import Path

#joblib 
from joblib import Parallel, delayed, parallel_backend

#scikit-learn
from sklearn.ensemble import RandomForestRegressor

from sklearn.preprocessing import MinMaxScaler
from sklearn.base import clone
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
#import dense, conv1d, flatten from keras
from keras.models import Sequential
from keras.layers import Dense, Conv1D, Flatten, Dropout, GlobalMaxPooling1D, BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping


#statsmodels
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.statespace.varmax import VARMAX
from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.graphics.utils import create_mpl_fig
from statsmodels.tsa.stattools import adfuller

#tqdm  
from tqdm import tqdm

#import functions from modules
from helper_functions import prepare_forecast_data, prepare_rf_lagged_only, invert_transformation, prepare_rf_lagged_only, time_delay_embedding, interpolate_monthly_from_annual, add_lags
from models import naive_forecast, recursive_forecast_rf_strict
import traceback



def get_forecast(config_obj):
    '''
    Given a config object, this function will train a model and return a forecast for the next 12 months.
    The function will also return a chart of the forecast and a chart of the test set results.

    Parameters
    ----------
    config_obj : config object
        A config object that contains the model, data, country, target, params for the model.

    Returns
    -------
    forecast_object : forecast object
        A forecast object that contains the model, data, country, target, params for the model, the forecast, the plot of the original data, and the test set results.

    '''


    #get parameters from config object
    params = config_obj['params']
    indicator = config_obj['indicator']
    country = config_obj['country']
    model_name = config_obj['model']
    data = config_obj['data']
    target = config_obj['target']
    predictors = config_obj['predictors']
    log_transform = config_obj['log_transform']
    walkforward = config_obj['walkforward']
    pt = config_obj['pt']

    #get today's date for saving forecast
    today = dt.datetime.today().strftime('%Y-%m-%d')
    today_month = dt.datetime.today().strftime('%B')
    today_year = dt.datetime.today().strftime('%Y')


    #train model and get metrics
    #try:
    print("Training ", model_name.__name__, "for ", country	)
    # try:
    #   if model_name.__name__ == 'random_forest_w_lags':
    #     #prepare data for random forest
    #     X, y = prepare_rf_lagged_only(data, lags=11, horizon=12)
    #     data = pd.concat([X, y], axis=1)
    #     trained_model = model_name(data, country, target)
    #     mod_name = trained_model.meta['model_name']
    #     naive = naive_forecast(data.iloc[-1], country, target, predictors, mod_name, log_transform, pt = pt)
    #   else:
    #     mod_name = trained_model.meta['model_name']
    #     trained_model = model_name(data, country, target, params = params, predictors = predictors, walkforward = walkforward, log_transform = log_transform, pt = pt)
    #     naive = naive_forecast(data, country, target, predictors, mod_name, log_transform, pt = pt)


        
    # except Exception as e:
    #   print("An error occurred while training the model", model_name.__name__, "for: ", country, e)
    #   #print line of error 
    #   import traceback
    #   traceback.print_exc()
    #   return None

    try:
      if model_name.__name__ == 'random_forest_w_lags':
        #prepare data for random forest
        # X, y = prepare_rf_lagged_only(data, lags=11, horizon=12)
        # data = pd.concat([X, y], axis=1)
        trained_model = model_name(data, country, target)

        mod_name = trained_model.meta['model_name']

      else:
        mod_name = trained_model.meta['model_name']
        trained_model = model_name(data, country, target, params = params, predictors = predictors, walkforward = walkforward, log_transform = log_transform, pt = pt)
        naive = naive_forecast(data, country, target, predictors, mod_name, log_transform, pt = pt)
      
    except:
      print("An error occurred while training the model", model_name.__name__, "for: ", country, sys.exc_info()[0])
      #traceback 
      traceback.print_exc()
      #print the line where the error is 
      print("Error occurred at line: ", sys.exc_info()[-1].tb_lineno)
      return None


    #prepare data for use in model 
    X, y = prepare_forecast_data(data, target)

    # set frequency of data to monthly and fill in missing values
    if X is not None:
      X = X.asfreq("MS")  
      y = y.asfreq("MS")

      X = X.fillna(method='ffill', limit=12)
      y = y.fillna(method='ffill', limit=12)

    else:
      y = y.asfreq("MS")  
      y = y.fillna(method='ffill', limit=12)
    
    #get date of last observation
    last_obs = y.index[-1]

    #set start of forecast to be 1 month after last observation
    forecast_start = last_obs + pd.DateOffset(months = 1)
    
    #fit model and forecast for next 12 months depending on model passed in config object
    if(model_name.__name__ == 'arima'):
      mod_name ='ARIMA'
      model = ARIMA(y, order=params[0], seasonal_order=params[1],enforce_stationarity=False,
                            enforce_invertibility=False)
      try:
        # code that might raise an exception
        print("Fitting ARIMA model for: ", country	)
        model_fit = model.fit()
      except:
        # code to handle the exception
        print("An error occurred while fitting the model: ", sys.exc_info()[0])
        return None
      
      forecast = model_fit.forecast(steps=16)
      conf_int = model_fit.get_forecast(steps=16).conf_int()
      conf_int.columns = ['lower', 'upper']



      if log_transform:
        #reverse power transform
        forecast_arr = pt.inverse_transform(np.array(forecast).reshape(-1,1)).squeeze()

        #forecast_arr = (20 * np.exp(forecast) +1)  / (1 + np.exp(forecast))
        forecast = pd.Series(forecast_arr, index = forecast.index,name="predicted_mean")

        #same for confidence interval
        conf_int = model_fit.get_forecast(steps=16).conf_int()
        conf_int.columns = ['lower', 'upper']

        conf_int['lower'] = pt.inverse_transform(np.array(conf_int['lower']).reshape(-1,1)).squeeze()
        conf_int['upper'] = pt.inverse_transform(np.array(conf_int['upper']).reshape(-1,1)).squeeze()

        #conf_int['lower'] = (20 * np.exp(conf_int['lower']) +1)  / (1 + np.exp(conf_int['lower']))
        #conf_int['upper'] = (20 * np.exp(conf_int['upper']) +1)  / (1 + np.exp(conf_int['upper']))

          
    elif(model_name.__name__ == 'sarimax'): 
      #prepare data again
      data = data.dropna()
      X, y = prepare_forecast_data(data, target)

      # set frequency of data to monthly and fill in missing values
      if X is not None:
        X = X.asfreq("MS")  
        y = y.asfreq("MS")

        X = X.fillna(method='ffill', limit=12)
        y = y.fillna(method='ffill', limit=12)

      else:
        y = y.asfreq("MS")  
        y = y.fillna(method='ffill', limit=12)
      
      
      
      mod_name ='SARIMAX'
      
      #########
      #do arima forecast of X and use as exogenous variables for sarimax
      ########
      future_exog = pd.DataFrame()      

      with open(r'C:\Users\tfarotimi\Documents\CEPAL\Inflation Forecasting\dat\forecast_params.yaml', 'r') as file:
          parameters = yaml.load(file, Loader=yaml.FullLoader)

      #check if country exists in predictors_params.yaml
      pred_param_alias = country[0:3]

      #check the country code (first 3 letters) in each key in the exogenous variables params file, if it exists, use the params for that key to fit the model for the exogenous variable, else run auto arima and save params to yaml file
      for pk in parameters.keys():
        if pred_param_alias in pk:
          pred_param_country = pk
        else:
          pred_param_country = pred_param_alias
      
      #check if country exists in predictors_params.yaml
      if pred_param_country in parameters:
         pass;
      else:
         parameters[pred_param_country] = {}
      
      for i in predictors:
            #check if i exists in predictors_params.yaml

          
            if i in parameters[pred_param_country]:
                #if it exists, use params from yaml file to fit arima
                exog_params = parameters[pred_param_country][i]['params']
                exog_model = ARIMA(X[i], order=exog_params['order'], seasonal_order=exog_params['seasonal_order'], enforce_stationarity=False, enforce_invertibility=False).fit()
            
            #special case for global variables (change "YoY" to "global"), if they exist once in yaml file, they exist for all countries
            elif "YoY" in i:
                print ("found parameters for ", i)
                exog_params=parameters['ARG_raw'][i]['params']
                exog_model = ARIMA(X[i], order=exog_params['order'], seasonal_order=exog_params['seasonal_order'], enforce_stationarity=False, enforce_invertibility=False).fit()

            else:
                
                #run auto arima and predict 12 months of X[i] with best params
                print("Running auto arima for " + i + " in " + country)
                best_arima = pmd.auto_arima(X[i], start_p=0, start_q=0,
                                        test= 'kpss',   # use kpss to find optimal 'd'
                                        max_p=3,  max_q=3,
                                        start_P = 0, start_Q=0,
                                        max_P = 3,  max_Q = 3,
                                        seasonal = True,
                                        trace=False,
                                        m=12,             
                                        stepwise=True,
                                        scoring='mse')
            
                exog_model = best_arima.fit(X[i])
               

                #add params to yaml file for the country if it exists, else create new entry
                if pred_param_country in parameters:
                    parameters[pred_param_country][i] = {'params': best_arima.get_params()}
                else:
                    parameters[pred_param_country] = {i: {'params': best_arima.get_params()}}

                #write to yaml file
                with open(r'C:\Users\tfarotimi\Documents\CEPAL\Inflation Forecasting\dat\forecast_params.yaml', 'w') as file:
                    yaml.dump(parameters, file)
            
            #predict 12 months of X[i] with best params
            try:
              future_exog[i] = exog_model.forecast(12)

            except:
              future_exog[i] = exog_model.predict(12)

      #fit sarimax model and forecast using forecasted exogenous variables
      try:
        model = SARIMAX(y, order=params[0], seasonal_order=params[1],enforce_stationarity=False,
                            enforce_invertibility=False, innovations = 't')
        try:
          print ("Fitting SARIMAX model for forecasting ", country)
          model_fit = model.fit()
          forecast = model_fit.forecast(steps=16, exog=future_exog)

        except:
          print("An error occurred while fitting " + model_name.__name__ + " for " + country + " - " + sys.exc_info()[0])
          return None

        
        if log_transform:
          #reverse power transform
          forecast_arr = pt.inverse_transform(np.array(forecast).reshape(-1,1)).squeeze()
          forecast = pd.Series(forecast_arr, index = forecast.index, name='predicted_mean')
            


          #same for confidence interval
          conf_int = model_fit.get_forecast(steps=16, exog=future_exog).conf_int()
          conf_int['lower'] = pt.inverse_transform(np.array(conf_int['lower']).reshape(-1,1)).squeeze()
          conf_int['upper'] = pt.inverse_transform(np.array(conf_int['upper']).reshape(-1,1)).squeeze()



      except:
        if (X is None):
          print('No exogenous variables in data. Check data for exogenous variables.')

      #create index of 12 months from forecast start for forecast
      forecast.index = pd.date_range(forecast_start, periods = 12, freq = 'MS')

    
    elif (model_name.__name__ == 'varmax'):
      mod_name ='VARMAX'
      varmax_y = pd.concat([y, X], axis = 1)
      p = trained_model.meta['params']['order'][0]
      q = trained_model.meta['params']['order'][2]
      d = trained_model.meta['params']['order'][1]

      P = trained_model.meta['params']['seasonal_order'][0]
      Q = trained_model.meta['params']['seasonal_order'][2]
      D = trained_model.meta['params']['seasonal_order'][1]

      params = trained_model.meta['params']['order'] + trained_model.meta['params']['seasonal_order']

      if d == 0:
        model = VARMAX(varmax_y,order=(p,q), seasonal_order=(P,D,Q,12))
      else:
        model = VARMAX(varmax_y,order=(p,q), seasonal_order=(P,D,Q,12), trend='n')
        
      try:
          print("Fitting VARMAX model for forecasting ", country)
          model_fit = model.fit()
      except:
          print("An error occurred while fitting VARMAX model: ", "\n", sys.exc_info()[0])
          return None
      
      forecast = model_fit.get_forecast(steps=16).predicted_mean.iloc[:,0]

      if log_transform:
        #reverse power transform
        forecast_arr = pt.inverse_transform(np.array(forecast).reshape(-1,1)).squeeze()
        forecast = pd.Series(forecast_arr, index = forecast.index)

        #same for confidence interval
        conf_int = pd.DataFrame()
        conf_int['lower'] = model_fit.get_forecast(steps=16).conf_int().iloc[:,0]
        conf_int['upper'] = model_fit.get_forecast(steps=16).conf_int().iloc[:,4]

        conf_int['lower'] = pt.inverse_transform(np.array(conf_int['lower']).reshape(-1,1)).squeeze()
        conf_int['upper'] = pt.inverse_transform(np.array(conf_int['upper']).reshape(-1,1)).squeeze()

    elif (model_name.__name__ == 'var_model'):

      forecast = pd.DataFrame()

      mod_name ='VAR'
      var_Y = pd.concat([y, X], axis = 1)
      p = trained_model.meta['params']
      model_fit = trained_model.meta['model']
      diff = trained_model.meta['diff']
      differenced = trained_model.meta['differenced']
      irf = trained_model.meta['irf']
      fecv = trained_model.meta['fecv']
      params = p

      #create index of 12 months from forecast start for forecast
      forecast.index = pd.date_range(forecast_start, periods = 12, freq = 'MS')

      #fit model
      if diff == 0:
        forecast_input = var_Y.values[-p:]
        preds = model_fit.forecast(forecast_input, steps=16)[:,0]

        predictions = pd.DataFrame(preds)

        conf_int = pd.DataFrame()
        conf_int['lower'] = model_fit.forecast_interval(var_Y.values, steps=16)[1][:,0]
        conf_int['upper'] = model_fit.forecast_interval(var_Y.values, steps=16)[2][:,0]
        conf_int.index = forecast.index
      elif diff == 1:
        forecast_input = differenced.values[-p:]
        preds = pd.DataFrame(model_fit.forecast(forecast_input, steps=16),index = forecast.index, columns = var_Y.columns +"_1d")

        predictions = invert_transformation(var_Y, preds, diff = diff)

        conf_int = pd.DataFrame()
        conf_int['lower'] = model_fit.forecast_interval(differenced.values, steps=16)[1][:,0]
        conf_int['upper'] = model_fit.forecast_interval(differenced.values, steps=16)[2][:,0]

        #dedifference conf_int
        conf_int['lower'] = y.iloc[-1] + conf_int['lower'].cumsum()
        conf_int['upper'] = y.iloc[-1] + conf_int['upper'].cumsum()
        conf_int.index = forecast.index
      elif diff == 2:
        forecast_input = differenced.values[-p:]
        preds = pd.DataFrame(model_fit.forecast(forecast_input, steps=16),index = forecast.index, columns = var_Y.columns +"_2d")

        predictions = invert_transformation(var_Y, preds, diff = diff)

        conf_int = pd.DataFrame()
        conf_int['lower'] = model_fit.forecast_interval(differenced.values, steps=16)[1][:,0]
        conf_int['upper'] = model_fit.forecast_interval(differenced.values, steps=16)[2][:,0]

        conf_int['lower'] = y.iloc[-1] + (y.iloc[-1]-y.iloc[-2]) + conf_int['lower'].cumsum()
        conf_int['upper'] = y.iloc[-1] + (y.iloc[-1]-y.iloc[-2]) + conf_int['upper'].cumsum()
        conf_int.index = forecast.index

      elif diff == 3: 
        forecast_input = differenced.values[-p:]
        preds = pd.DataFrame(model_fit.forecast(forecast_input, steps=16),index = forecast.index, columns = var_Y.columns +"_3d")

        predictions = invert_transformation(var_Y, preds, diff = diff)

        conf_int = pd.DataFrame()
        conf_int['lower'] = model_fit.forecast_interval(differenced.values, steps=16)[1][:,0]
        conf_int['upper'] = model_fit.forecast_interval(differenced.values, steps=16)[2][:,0]

        conf_int['lower'] = y.iloc[-1] + (y.iloc[-1]-y.iloc[-2]) + (y.iloc[-2]-y.iloc[-3]) + conf_int['lower'].cumsum()
        conf_int['upper'] = y.iloc[-1] + (y.iloc[-1]-y.iloc[-2]) + (y.iloc[-2]-y.iloc[-3]) + conf_int['upper'].cumsum()
        conf_int.index = forecast.index
      
      elif diff == 4:
        forecast_input = differenced.values[-p:]
        preds = pd.DataFrame(model_fit.forecast(forecast_input, steps=16),index = forecast.index, columns = var_Y.columns +"_4d")

        predictions = invert_transformation(var_Y, preds, diff = diff)

        conf_int = pd.DataFrame()
        conf_int['lower'] = model_fit.forecast_interval(differenced.values, steps=16)[1][:,0]
        conf_int['upper'] = model_fit.forecast_interval(differenced.values, steps=16)[2][:,0]

        conf_int['lower'] = y.iloc[-1] + (y.iloc[-1]-y.iloc[-2]) + (y.iloc[-2]-y.iloc[-3]) + (y.iloc[-3]-y.iloc[-4]) + conf_int['lower'].cumsum()
        conf_int['upper'] = y.iloc[-1] + (y.iloc[-1]-y.iloc[-2]) + (y.iloc[-2]-y.iloc[-3]) + (y.iloc[-3]-y.iloc[-4]) + conf_int['upper'].cumsum()
        conf_int.index = forecast.index
    
    elif (model_name.__name__ == 'random_forest_w_lags'):
      mod_name = 'Random Forest w/ Lags'
      # Recreate the lagged data for forecasting
      full_data = data.copy()
      forecast_index = pd.date_range(forecast_start, periods=12, freq='MS')

      # Create lags for the full data
      lagged_data = []
      for col in full_data:
          col_df = time_delay_embedding(full_data[col], n_lags=12, horizon=0)
          lagged_data.append(col_df)

      full_data = pd.concat(lagged_data, axis=1).dropna()
      full_data.index = pd.to_datetime(full_data.index)
      full_data.index = full_data.index.to_period('M').to_timestamp()
      #ensure index freq is M
      full_data = full_data.asfreq('MS')
      # end match
      full_data.index.name = 'Period'
  
      full_data["trend"] = np.arange(len(full_data))                       # linear time index
      full_data["month"] = full_data.index.month.astype("category")        # seasonality dummies
      full_data = pd.get_dummies(full_data, columns=["month"], drop_first=True)

    # 4)  rolling statistics (mean & std) ---------------------------------------
      roll_windows = [3, 6]  # rolling windows in months
      for w in roll_windows:
          full_data[f"roll{w}_mean"] = full_data['YoY Increase Inflation(t)'].rolling(w).mean()
          full_data[f"roll{w}_std"]  = full_data['YoY Increase Inflation(t)'].rolling(w).std()

      # 5)  YoY and MoM percentage changes ----------------------------------------
      full_data["mom"] = full_data['YoY Increase Inflation(t)'].pct_change()                # month-on-month
      full_data["yoy"] = full_data['YoY Increase Inflation(t)'].pct_change(12)              # year-on-year

      full_data = full_data.dropna()  # drop rows with NaN values after feature engineering

      exog_targets = {"FEDFUNDS(t-0)": {'2025': 4.1, '2026': 3.1},
                "POILBREUSDM(t-0)": {'2025': 67.67, '2026': 63.288},
                "PFOODINDEXM(t-0)": {'2025': 128.329, '2026': 128.404}
      }

      try:
        exog_data = pd.read_excel(r"C:\Users\tfarotimi\Documents\CEPAL\Inflation Forecasting\dat\exog_lagged_data.xlsx", index_col=0)
        #standardize exog_data with StandardScaler 
        

        exog_data.index = pd.to_datetime(exog_data.index)
        exog_data.index = exog_data.index.to_period('M').to_timestamp()  # Convert to timestamp
        exog_data.index.name = 'Period'

        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        exog_data = pd.DataFrame(scaler.fit_transform(exog_data), index=exog_data.index, columns=exog_data.columns)
        print("exog_data shape", exog_data.shape)

        print("last index of exog_data", exog_data.index.max())

        future_exog = pd.DataFrame()

        for col in exog_data.columns:
          if '(t-0)' in col:
              col_targets = exog_targets[col]
              col_monthly = interpolate_monthly_from_annual(exog_data[col], col_targets)
              col_monthly.name = exog_data[col].name
              future_exog = pd.concat([future_exog, add_lags(col_monthly, 3)], axis=1)

          




        #print indexes

       

        # Optional: trim to shared range
        start = max(full_data.index.min(), exog_data.index.min())
        end = min(full_data.index.max(), exog_data.index.max())
        full_data = full_data.loc[(full_data.index >= start) & (full_data.index <= end)]

        print("LAST FULL_DATA INDEX", full_data.index.max())
        future_start = full_data.index.max() + pd.DateOffset(months=1)
        print("future_start", future_start)
        future_exog = future_exog[future_start:future_start + pd.DateOffset(months=11)]  # Keep only the last 12 months for forecasting
        future_exog.index = pd.to_datetime(future_exog.index).to_period("M").to_timestamp()
        future_exog.index.name = 'Period'

        # Merge known exogenous variables with full_data
        full_data = full_data.merge(exog_data, how='left', left_index=True, right_index=True)
        #add future_exog
        # Final sanity check
        print(full_data.shape, "all _data, exogenous lagged data merged successfully.")
      except FileNotFoundError:
        print("No exogenous lagged data found. Proceeding with available data.")

      # Ensure the index is a datetime index
      full_data.index = pd.to_datetime(full_data.index)
      full_data.index = full_data.index.to_period('M').to_timestamp(how='start')

      print("full_data shape after feature engineering: ", full_data.shape)


      rf_model = trained_model.meta['model']
      X_train = trained_model.meta['X_train']
      y_train = trained_model.meta['y_train']
      target = 'YoY Increase Inflation(t)'
      preds = []
      lower_bounds = []
      upper_bounds = []



      forecast  = recursive_forecast_rf_strict(rf_model, full_data[target], future_exog, 12)
      print(forecast)
      print("forecast index")

      #add 'params' to forecast object 
      #make random forest best params
      params = trained_model.meta['params']


      # Create a DataFrame for the forecast 
      forecast.name = 'predicted_mean'

      # forecast.index = forecast_index 
      #debug 
      


      


      bias_corrected = "tbd" # shape (12,)
      # print("Bias corrected forecast for ", country, " using Random Forest w/ Lags: ", bias_corrected)

      #forecast = pd.Series(point_pred, index=forecast_index, name='predicted_mean')

      n_bootstraps = 100
      h = 12  # forecast horizon
      bootstrap_preds = []

      print ("fails in bootstrap_preds")
      bootstrap_preds = parallel_bootstrap_forecasts(n_bootstraps, X_train, y_train, rf_model, full_data, future_exog)

      # Confidence intervals
      print("conf ints")
      lower_bounds = np.percentile(bootstrap_preds, 2.5, axis=0)
      upper_bounds = np.percentile(bootstrap_preds, 97.5, axis=0)
      point_forecast = np.median(bootstrap_preds, axis=0)  # or mean

      last_date = trained_model.index[-1]

      print("lower_bounds:", lower_bounds)
      print("upper_bounds:", upper_bounds)

      conf_int = pd.DataFrame({
          'lower': lower_bounds,
          'upper': upper_bounds
      }, index=forecast_index)

      # Plot
      plt.fill_between(range(1, h+1), lower_bounds, upper_bounds, alpha=0.25, label="95% CI")
      plt.plot(range(1, h+1), point_forecast, marker="o", label="Forecast")
      plt.title("95% Bootstrap Forecast Intervals")
      plt.xlabel("Horizon (months ahead)")
      plt.legend()
      plt.show()




      
    elif (model_name.__name__ == 'cnn_model'):

      mod_name ='CNN'
      #prepare data for convolutional neural network
    # Normalize the data
      scaler = MinMaxScaler(feature_range=(0, 1))
      norm_data = scaler.fit_transform(data['YoY Increase Inflation'].values.reshape(-1, 1))
      # Convert to DataFrame for easier manipulation
      norm_data = pd.DataFrame(norm_data, columns=['YoY Increase Inflation'], index=data.index)
      # Define the number of lags

      #use adf uller to determine the number of lags
      n_lags = 24
      # Define the train size, everything but last 12 months
      train_size = len(norm_data) - 12



      #import max1dpooling 
      from keras.layers import MaxPooling1D

      # Prepare supervised learning samples from the full normalized data before splitting
      all_values = norm_data['YoY Increase Inflation'].values
      X_all, y_all = [], []
      for i in range(n_lags, len(all_values)):
          
          if i + 12 <= len(all_values):
            X_all.append(all_values[i - n_lags:i])
            y_all.append(all_values[i:i + 12])


      X_all = np.array(X_all)
      y_all = np.array(y_all)  # now shape: (samples, 12) 

        # Reshape for CNN input
      X_all = X_all.reshape((X_all.shape[0], X_all.shape[1], 1))


      from keras.layers import InputLayer
      from tcn import TCN

      
      

      # Build and train CNN model
      input_shape = (X_all.shape[1], 1)  # Define input shape for the InputLayer
      output_steps = 12           # Define output_steps, or set as needed
      model = Sequential()
      model.add(InputLayer(input_shape=input_shape))
      model.add(TCN(nb_filters=64, kernel_size=3, dilations=[1, 2, 4, 8], activation='relu', dropout_rate=0.2))
      model.add(Dense(50, activation='relu'))
      model.add(Dense(output_steps))  # output_steps = number of forecast steps (e.g., 12)
      model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

      model.fit(X_all, y_all, epochs=200, verbose=0, batch_size=10)

    # 3. Custom function to enable MC Dropout at inference time
      import tensorflow as tf
      def predict_mc(model, X, n_iter=100):
          preds = tf.stack([model(X, training=True) for _ in range(n_iter)], axis=0)
          mean = tf.reduce_mean(preds, axis=0)
          std = tf.math.reduce_std(preds, axis=0)
          print(mean)
          print(std)
          # Ensure tensors are converted to numpy arrays
          # mean = mean.numpy() if hasattr(mean, 'numpy') else np.array(mean)
          # std = std.numpy() if hasattr(std, 'numpy') else np.array(std)
          lower = mean - 1.96 * std
          upper = mean + 1.96 * std
          return mean[0], lower[0], upper[0]  # Return the first (and only) sample

      # 4. Forecast with MC Dropout
      forecast_input = X_all[-1].reshape(1, n_lags, 1)
      mean_forecast, lower_ci, upper_ci = predict_mc(model, forecast_input, n_iter=100)

      # 5. Print results
      # print("Forecast:", mean_forecast)
      # print("95% CI lower:", lower_ci)
      # print("95% CI upper:", upper_ci)




      #get forecast confidence interval for cnn model

      #reshape forecast to 2d array
      forecast = np.array(mean_forecast).reshape(-1, 1)


      # Rescale predictions back to original scale
      forecast = scaler.inverse_transform(forecast)
      lower_ci = scaler.inverse_transform(np.array(lower_ci).reshape(-1, 1))
      upper_ci = scaler.inverse_transform(np.array(upper_ci).reshape(-1, 1))
   
      # Create a date range for the forecast
      forecast_index = pd.date_range(start=forecast_start, periods=12, freq='MS')
      forecast = pd.Series(forecast.flatten(), index=forecast_index, name='predicted_mean')
      # Create a DataFrame for the forecast
      conf_int = pd.DataFrame(index=forecast_index)
      conf_int['lower'] = lower_ci
      conf_int['upper'] = upper_ci
      conf_int.index = forecast.index
      #conf_int = pd.DataFrame(index=forecast_index)



      
      if log_transform:
        #reverse power transform
        forecast_arr = pt.inverse_transform(np.array(forecast).reshape(-1,1)).squeeze()
        #forecast_arr = (20 * np.exp(forecast) +1)  / (1 + np.exp(forecast))



        forecast = pd.Series(forecast_arr, index = forecast.index, name='predicted_mean')

        #calculate confidence interval
        conf_int['lower'] = pd.Series(pt.inverse_transform(np.array(conf_int['lower']).reshape(-1,1)).squeeze(), index = forecast.index, name='Predicted')
        conf_int['upper'] = pd.Series(pt.inverse_transform(np.array(conf_int['upper']).reshape(-1,1)).squeeze(), index = forecast.index, name='Predicted')
        #conf_int['lower'] = (20 * np.exp(conf_int['lower']) +1)  / (1 + np.exp(conf_int['lower']))
        #conf_int['upper'] = (20 * np.exp(conf_int['upper']) +1)  / (1 + np.exp(conf_int['upper']))


    #get naive if random forest   
    if (mod_name == 'Random Forest w/ Lags'):
      #get naive forecast
      rf_data = {}
      rf_data['X-train'] = trained_model.meta['X_train']
      rf_data['y-train'] = trained_model.meta['y_train']
      rf_data['X-test'] = trained_model.meta['X_test']
      rf_data['y-test'] = trained_model.meta['y_test']
      naive = naive_forecast(rf_data, country, target, predictors, mod_name, log_transform, pt = pt)

      # #add naive forecast to forecast
      # forecast = pd.concat([naive, forecast], axis=0)
      # conf_int = pd.concat([naive, conf_int], axis=0)
    
    #create forecast chart
    if (mod_name in ['SARIMAX', 'ARIMA', 'VARMAX','VAR', 'CNN', 'Random Forest w/ Lags']):  
      p = forecast
      #get last 12 months of y
      all_Y = (y[-12:])

      #reverse log of y if log_transform is True
      if log_transform:
        all_Y_arr = pt.inverse_transform(np.array(all_Y).reshape(-1,1)).squeeze()
        #all_Y_arr = (20 * np.exp(all_Y) +1)  / (1 + np.exp(all_Y))
        all_Y = pd.Series(all_Y_arr, index = all_Y.index)

      #add forecast to all_Y
      #if bias corrected, add bias corrected forecast to all_Y
      if mod_name == 'Random Forest w/ Lags':
        #if bias corrected, add bias corrected forecast to all_Y
        if bias_corrected == "tbd":
          all_Y = pd.concat([all_Y, forecast], axis= 0)
        else:
          all_Y = pd.concat([all_Y, forecast], axis= 0)
          # bias_vec is an ndarray shape (1, 12)  ->   flatten to (12,)
          bias_vec = trained_model.meta['bias_vec']
          bias_series = pd.Series(
          bias_vec.flatten(),          # length 12
          index=conf_int.index         # e.g. ['t+1','t+2', … 't+12']
          )

        # now add row-wise: each horizon’s lower & upper get shifted by its bias
          conf_int = conf_int.add(bias_series, axis=0)

       
      else:
        #if not bias corrected, just add forecast to all_Y
        all_Y = pd.concat([all_Y, forecast], axis=0)


      print("all_Y:", all_Y)
      print("forecast:", forecast)
      fig3 = go.Figure()
      fig3.add_trace(go.Scatter(x = all_Y.index, y = all_Y, name = target,showlegend=False))
      #fig3.update_layout(title = country + " - " + indicator + " - " + model_name + " 12 months forecast ", xaxis_title = 'Date', yaxis_title = target, height = 600, width = 1400)
      fig3.update_layout(title = country, xaxis_title = 'Date', yaxis_title = target, height = 600, width = 1400)
      
      #add line for current date
      fig3.add_shape(type="line", yref='paper', x0=y.index[-1], y0=0, x1=y.index[-1], y1=1, line=dict(color="Red",width=1, dash="dot")) 
      
      #add label for each value of forecast
      for i in range(0, 12):
        fig3.add_annotation(
          xref='x1',
          yref='y1',
          yanchor="bottom",
          borderpad=1,
          x=all_Y.index[-12+i],
          y=all_Y.iloc[-12+i],
          font=dict(size=12),
          bgcolor="white",
          bordercolor='black',
          borderwidth=1,
          text = str(round(all_Y.iloc[-12 + i], 2)),
          showarrow=False
        )

      #
      if (mod_name in ['ARIMA', 'VARMAX']):
        p = model_fit.get_prediction(start = forecast_start, end = forecast_start + pd.DateOffset(months = 11))
      elif (mod_name == 'SARIMAX'):
        p = model_fit.get_prediction(start = forecast_start, end = forecast_start + pd.DateOffset(months = 11), exog=future_exog)
      elif (mod_name == 'VAR'):
        p = forecast
      elif (mod_name == 'CNN'):
        p = forecast
      elif (mod_name == 'Random Forest w/ Lags'):

        p = pd.DataFrame(forecast, index=forecast_index, columns=['Predicted'])
        p.index.name = 'Date'
        p = p.rename(columns={'Predicted': 'predicted_mean'})
    
     

      #save confidence interval to dataframe
      if mod_name == 'VARMAX':
        target_cols = [col for col in p.conf_int().columns if varmax_y.columns[0] in col]
        conf_int = p.conf_int()[target_cols]

        

        if log_transform:
          p.predicted_mean = pd.Series(pt.inverse_transform(np.array(p.predicted_mean.iloc[:,0]).reshape(-1,1)).squeeze(), index = p.predicted_mean.index, name='Predicted')

          #calculate confidence interval
          conf_int['lower'] = pd.Series(pt.inverse_transform(np.array(conf_int.iloc[:,0]).reshape(-1,1)).squeeze(), index = p.predicted_mean.index, name='Predicted')
          conf_int['upper'] = pd.Series(pt.inverse_transform(np.array(conf_int.iloc[:,1]).reshape(-1,1)).squeeze(), index = p.predicted_mean.index, name='Predicted')

          #concatenate lower and upper
          conf_int = pd.concat([conf_int['lower'], conf_int['upper']], axis = 1)
          conf_int.columns = ['lower', 'upper']
        

          


      
      
      #reverse log of predicted mean and conf_int if log_transform is True
     

      #fill in fig3 with confidence intervals
        
      
      fig3.add_trace(go.Scatter(x = conf_int.index, y = conf_int['upper'], showlegend=False))
      fig3.add_trace(go.Scatter(x = conf_int.index, y = conf_int['lower'], name = '95% confidence interval',  fill = 'tonexty',showlegend=False))
  


      

      #get metrics for test set evaluation
      rmse = trained_model.meta['rmse']
      r2 = trained_model.meta['r2_score']
      aic = trained_model.meta['aic'] if mod_name != 'Random Forest w/ Lags' else None
      mape = trained_model.meta['mape']

  

      #save image
      CHART_PATH = r"reports\\" + today_year + "/" + today_month + "/" + today + "/" + country + "/" + mod_name + "/" + country +" _"+ indicator + '_forecastchart.png'
      parent_dir = Path(CHART_PATH).parent.absolute()
      
      #create directory if it doesn't exist
      if not(os.path.exists(parent_dir)):
        os.makedirs(parent_dir)
       
      pio.write_image(fig3, file=CHART_PATH)

      #add border to image
      img = Image.open(CHART_PATH)
      bordered_img = ImageOps.expand(img, border=1, fill='black')
      bordered_img.save(CHART_PATH)

      #create test set evaluation results chart 
      test_chart = go.Figure()

      if mod_name == 'Random Forest w/ Lags':
        features = trained_model.meta['feature_names']
        importances = trained_model.meta['feature_importances']
        #first 12 columns of trained_model are actual values, last 12 columns are predicted value
# Create new columns properly
#assign index
        #replace index    


        x_index = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=12, freq='MS')
        y_pred_values = trained_model['Predicted']
        y_pred_values.index = x_index
        y_act_values = trained_model['Actual']
        y_act_values.index = x_index

        # Create dataframe
        importance_df = pd.DataFrame({
            'Feature': features,
            'Importance': importances
        }).sort_values(by='Importance', ascending=False)

        # Create bar chart for feature importance
        importance_chart = go.Figure(data=[go.Bar(x=importance_df['Feature'], y=importance_df['Importance'])])
        importance_chart.update_layout(title_text='Feature Importance for Random Forest Model', xaxis_title='Features', yaxis_title='Importance')
        #save image
        IMPORTANCE_CHART_PATH = r"C:\Users\tfarotimi\Documents\CEPAL\Inflation Forecasting\reports\\" + today_year + "/" + today_month + "/" + today + "/" + country + "/" + mod_name + "/" + country +" _"+ config_obj['indicator'] + "_Feature Importance.png"
        parent_dir = Path(IMPORTANCE_CHART_PATH).parent.absolute()
        if not(os.path.exists(parent_dir)):
          os.makedirs(parent_dir)
        pio.write_image(importance_chart, file=IMPORTANCE_CHART_PATH)
        #add border to image
        img = Image.open(IMPORTANCE_CHART_PATH)
        bordered_img = ImageOps.expand(img, border=1, fill='black')
        bordered_img.save(IMPORTANCE_CHART_PATH)


        

      else:
        x_index = trained_model['Predicted'].index
        #for random forest, use the predicted values from the forecast
        y_pred_values = trained_model['Predicted']
        y_act_values = trained_model['Actual']
        



      test_chart.add_trace(go.Scatter(x=x_index, y=y_pred_values, name='Predicted', line_color='blue'))
      test_chart.add_trace(go.Scatter(x=x_index, y=y_act_values, name='Actual', line_color='indianred'))
      #add trace for 20% error
      test_chart.add_trace(go.Scatter(x=x_index, y=y_act_values + (y_act_values * 0.2), name='20% Error', line=dict(color = 'mediumaquamarine', width=4, dash='dot'),mode = 'lines',showlegend=False))
      #add trace for -20% error
      test_chart.add_trace(go.Scatter(x=x_index, y=y_act_values - (y_act_values * 0.2), name='+/-20% Error', line=dict(color = 'mediumaquamarine',width=4, dash='dot'), mode = 'lines', fill="tonexty"))



      #order traces in test_chart
      test_chart.data = (test_chart.data[2], test_chart.data[3],test_chart.data[1],test_chart.data[0])

      #add forecast['test_metrics'] as annotation on test_chart
      test_chart.add_annotation(
          x=0.5,
          y=0.9,
          text="RMSE: " + str(round(rmse,2)) + " | Naive RMSE: " + str(round(naive.meta['rmse'],2)) + " | MAPE: " + str(round(mape,2)) + "%",
          bordercolor="black",
          borderwidth=1,
          borderpad=4,
          bgcolor="white",
          showarrow=False,
          xref="paper",
          yref="paper"
      )


      #add  labels to test_chart
      test_chart.update_yaxes(title_text="Inflation YOY Increase (%)")
      test_chart.update_xaxes(title_text="Date")


      
      #save image
      TEST_CHART_PATH = r"C:\Users\tfarotimi\Documents\CEPAL\Inflation Forecasting\reports\\" + today_year + "/" + today_month + "/" + today + "/" + country + "/" + mod_name + "/" + country +" _"+ config_obj['indicator'] + "_Test Set Results.png"
      
      parent_dir = Path(TEST_CHART_PATH).parent.absolute()
      
      if not(os.path.exists(parent_dir)):
        os.makedirs(parent_dir)
       
      pio.write_image(test_chart, file=TEST_CHART_PATH)

      #add border to image
      img = Image.open(TEST_CHART_PATH)
      bordered_img = ImageOps.expand(img, border=1, fill='black')
      bordered_img.save(TEST_CHART_PATH)


      #create plot for data used for model parameter estimation
      orig_data_chart = go.Figure()

      orig_data_chart.add_trace(go.Scatter(x=y.index, y=y, name=y.name))

      orig_data_chart.update_layout(title_text= 'Appendix B: '+ country + ' Data Set Used For Model Parameter Estimation',width=800)

      #add forecast['test_metrics'] as annotation on test_chart
      orig_data_chart.add_annotation(
          x=0.5,
          y=0.9,
          text = "Mean: " + str(round(y.mean(),2)) + " | Median: " + str(round(y.median(),2)) + " | St Dev :" + str(round(y.std(),2)) + " | Min: " + str(round(y.min(),2)) + " | Max: " + str(round(y.max(),2)) + " | Kurtosis: " + str(round(y.kurtosis(),2)),
          bordercolor="black",
          borderwidth=1,
          borderpad=4,
          bgcolor="white",
          showarrow=False,
          xref="paper",
          yref="paper"
      )


      #add  labels to orig_data_chart
      orig_data_chart.update_yaxes(title_text="Inflation YOY Increase (%)")
      orig_data_chart.update_xaxes(title_text="Date")


    
      #save image
      ORIG_DATA_CHART_PATH = r"reports" + "/" + today_year + "/" + today_month + "/" + today + "/" + country + "/" + mod_name + "/" + country +" _"+ config_obj['indicator'] + "_Input Data Set.png"
      
      parent_dir = Path(ORIG_DATA_CHART_PATH).parent.absolute()
      
      if not(os.path.exists(parent_dir)):
        os.makedirs(parent_dir)
  

      pio.write_image(orig_data_chart, file=ORIG_DATA_CHART_PATH)

      #add border to image
      img = Image.open(ORIG_DATA_CHART_PATH)
      bordered_img = ImageOps.expand(img, border=1, fill='black')
      bordered_img.save(ORIG_DATA_CHART_PATH)

      #create and save distribution of residuals chart
      if mod_name in ['Random Forest w/ Lags']:
          residuals = trained_model.meta['residuals']
          residuals_chart = go.Figure()
          residuals_chart.add_trace(go.Histogram(x=np.array(residuals).flatten(), nbinsx=30))
          residuals_chart.update_layout(title_text='Distribution of Residuals - ' + country, xaxis_title_text='Residuals', yaxis_title_text='Count')
          RESIDUALS_CHART_PATH = r"reports" + "/" + today_year + "/" + today_month + "/" + today + "/" + country + "/" + mod_name + "/" + country +" _"+ config_obj['indicator'] + "_Residuals Distribution.png"
          pio.write_image(residuals_chart, file=RESIDUALS_CHART_PATH)

          img = Image.open(RESIDUALS_CHART_PATH)
          bordered_img = ImageOps.expand(img, border=1, fill='black')
          bordered_img.save(RESIDUALS_CHART_PATH)

      if mod_name in ['VAR', 'CNN', 'Random Forest w/ Lags']: 
         #plot autocorrelation for first variable in VAR model
        #plot_diagnostics = model_fit.plot_acorr(nlags=12, resid = True, linewidth = 8)
        plot_diagnostics = None
      else:
          plot_diagnostics = model_fit.plot_diagnostics()

      #create forecast object for creating excel report 
      forecast_object = {
                        'model': mod_name,
                        'data':data,
                        'params':params,	
                        'country':country,
                        'indicator':indicator,
                        'target':target,
                        'forecast':forecast,
                        'conf_int':conf_int,
                        'chart':CHART_PATH,
                        'test_chart':TEST_CHART_PATH,
                        'orig_data_chart':ORIG_DATA_CHART_PATH,
                        'importance_chart':IMPORTANCE_CHART_PATH if mod_name == 'Random Forest w/ Lags' else None,
                        'residuals_chart':RESIDUALS_CHART_PATH if mod_name == 'Random Forest w/ Lags' else None,
                        'test_predictions':y_pred_values,
                        'test_actual':y_act_values,
                        'test_metrics':{'RMSE':str(round(rmse,2)), 'Naive RMSE':str(round(naive.meta['rmse'],2)), 'AIC':str(round(aic,2)) if aic is not None else None, 'MAPE':str(round(mape,2)), 'Naive MAPE': str(round(naive.meta['mape'],2)), 'R2':str(round(r2,2))},
                        'model_summary':model_fit if mod_name in ['ARIMA', 'SARIMAX', 'VARMAX', 'VAR'] else None,
                        'residuals':model_fit.resid if mod_name in ['ARIMA', 'SARIMAX', 'VARMAX'] else None,
                        'rf_residuals':trained_model.meta['residuals'] if mod_name == 'Random Forest w/ Lags' else None,
                        'plot_diagnostics':plot_diagnostics or None, 
                        'predictors':predictors,
                        'naive_rmse':round(naive.meta['rmse'],2),
                        'log_transform':log_transform,
                        'irf': irf if mod_name == 'VAR' else None,
                        'fecv': fecv if mod_name == 'VAR' else None,
                        'bias_corrected': None, #bias_corrected if mod_name == 'Random Forest w/ Lags' else None,
                        'bias_vec': None, #trained_model.meta['bias_vec'] if mod_name == 'Random Forest w/ Lags' else None,
                        'rmse_ts': trained_model.meta['rmse_ts'] if mod_name == 'Random Forest w/ Lags' else None,
                        'bi_co_residuals': trained_model.meta['bi_co_residuals'] if mod_name == 'Random Forest w/ Lags' else None,
                        'ts_bias_corrected': trained_model.meta['y_hat_ts'] if mod_name == 'Random Forest w/ Lags' else None,
                        'shap_values': trained_model.meta['shap_values'] if mod_name == 'Random Forest w/ Lags' else None,
                        'exog_targets': exog_targets if mod_name == 'Random Forest w/ Lags' else None


                        }
      
      ##write forecast_object to json file   
      import json
      from datetime import datetime
      today_time = datetime.now().strftime("%H%M%S")
      forecast_object_path = r"reports\\" + today_year + "/" + today_month + "/" + today + "/" + country + "/" + mod_name + "/" + country +" _"+ config_obj['indicator'] + "_" + today_time + "_forecast_object.json"
      parent_dir = Path(forecast_object_path).parent.absolute()
      if not(os.path.exists(parent_dir)):
          os.makedirs(parent_dir)
      with open(forecast_object_path, 'w') as f:
          json.dump(forecast_object, f, indent=4, default=str)
      print("Forecast object saved to ", forecast_object_path)
      
      




      return forecast_object

def forecast_multistep_with_features(full_data, rf_model, steps=12):
    """
    Perform multi-step forecasting using a trained Random Forest model,
    updating lag features and other engineered features at each step.

    Parameters:
    - full_data: pd.DataFrame with feature columns including lagged variables and original inflation data.
    - rf_model: trained RandomForestRegressor model.
    - steps: int, number of future periods to forecast.

    Returns:
    - forecasts: list of predicted values.
    - full_data: extended DataFrame with appended predictions and features.
    """
    forecasts = []

    for _ in range(steps):
        # Step 1: Build X_input from last full_data row
        latest_row = full_data.iloc[-1:].copy()
        X_input = latest_row.drop(columns='YoY Increase Inflation(t)').values

        # Step 2: Predict next value
        point_pred = rf_model.predict(X_input)[0]
        forecasts.append(point_pred)

        # Step 3: Construct new row
        new_row = {}

        # Lag features: shift previous lags
        for i in range(1, 13):
            if i == 1:
                new_row[f'YoY Increase Inflation(t-{i})'] = full_data['YoY Increase Inflation(t)'].iloc[-1]
            else:
                new_row[f'YoY Increase Inflation(t-{i})'] = full_data[f'YoY Increase Inflation(t-{i-1})'].iloc[-1]

        # New prediction
        new_row['YoY Increase Inflation(t)'] = point_pred

        # Trend
        new_row['trend'] = full_data['trend'].iloc[-1] + 1

        # Month dummies
        next_month = (full_data.index[-1] + pd.DateOffset(months=1)).month
        for m in range(2, 13):
            new_row[f'month_{m}'] = 1 if next_month == m else 0

        # Rolling stats
        recent_values = full_data['YoY Increase Inflation(t)'].iloc[-6:].tolist() + [point_pred]
        new_row['roll3_mean'] = np.mean(recent_values[-3:])
        new_row['roll3_std'] = np.std(recent_values[-3:])
        new_row['roll6_mean'] = np.mean(recent_values)
        new_row['roll6_std'] = np.std(recent_values)

        # MoM and YoY
        prev_val = full_data['YoY Increase Inflation(t)'].iloc[-1]
        prev_12_val = full_data['YoY Increase Inflation(t)'].iloc[-12] if len(full_data) >= 12 else np.nan
        new_row['mom'] = (point_pred - prev_val) / prev_val if prev_val != 0 else 0
        new_row['yoy'] = (point_pred - prev_12_val) / prev_12_val if prev_12_val != 0 else 0

        # Step 4: Append to full_data
        new_index = full_data.index[-1] + pd.DateOffset(months=1)
        full_data.loc[new_index] = new_row

    return forecasts, full_data

def single_bootstrap_forecast(i, X_train, y_train, rf_model, full_data, future_exog):
  # Bootstrap sample
  block_size = 12  # e.g. 12 months
  n_blocks = len(X_train) // block_size

  X_blocks = []
  y_blocks = []

  for _ in range(n_blocks):
      start = np.random.randint(0, len(X_train) - block_size + 1)
      X_block = X_train.iloc[start:start + block_size]
      y_block = y_train.iloc[start:start + block_size]

      X_blocks.append(X_block)
      y_blocks.append(y_block)

  X_boot = pd.concat(X_blocks)
  y_boot = pd.concat(y_blocks)
  # Clone the model to avoid overwriting rf_model
  boot_model = clone(rf_model)
  boot_model.fit(X_boot, y_boot)

  # Forecast recursively
  full_series_boot = full_data['YoY Increase Inflation(t)'].copy()
  forecast_steps = recursive_forecast_rf_strict(boot_model, full_series_boot, future_exog, 12)
  return forecast_steps.values

def parallel_bootstrap_forecasts(n_bootstraps, X_train, y_train, rf_model, full_data, future_exog):
  results = Parallel(n_jobs=-1)(
      delayed(single_bootstrap_forecast)(i, X_train, y_train, rf_model, full_data, future_exog)
      for i in tqdm(range(n_bootstraps), desc="Bootstrapping")
  )
  print("finished bootstrapping ")
  return results