# This file contains the function that runs the monthly ARIMA forecast for each country. 
# It takes in the data for all countries and a list of countries to run the forecast for.
# It returns a list of forecasts for each country in the list of countries.
# Last update: 08.14.2023 by Inflation Forecasting Farotimi

# import libraries
import gc
from multiprocessing import process
import numpy as np
import yaml
from helper_functions import auto_grid_search, write_params_to_file, boxcox_inverse, boxcox_transform
from forecast import get_forecast
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import kpss
from sklearn.preprocessing import PowerTransformer
from tqdm import tqdm
from joblib import Parallel, delayed, parallel_backend

# Define the function to process each country
def process_country(key, data, model, model_name, parameters, log_transform, walkforward):
        print("Began processing country:", key)
        try:
            country_df = data[key].copy()

            #print country name
            print(f"Processing {key} with model {model_name}")

            if model_name == 'arima':
                country_df = country_df.iloc[:, 0:1]

            country_df.replace([np.inf, -np.inf], np.nan, inplace=True)
            country_df.dropna(subset=country_df.columns[0], inplace=True)

            pt = None
            if log_transform:
                pt = PowerTransformer(method='yeo-johnson')
                try:
                    pt.fit(country_df.iloc[:, 0].values.reshape(-1, 1))
                    country_df.iloc[:, 0] = pt.transform(country_df.iloc[:, 0].values.reshape(-1, 1))
                except:
                    pt = None
                    log_transform = False

                country_df.replace([np.inf, -np.inf], np.nan, inplace=True)
                country_df.dropna(subset=country_df.columns[0], inplace=True)

            start_date = country_df.index[0]
            end_date = country_df.index[-1]

            if model_name == 'random_forest_w_lags':
                params = 'TBD'
            else: 
                if key in parameters.keys():
                    params = parameters[key]['params']
                else:
                    if model_name == 'arima':
                        order, seasonal_order = auto_grid_search('arima', country_df)
                        params = (order, seasonal_order)
                        write_params_to_file(key, start_date, end_date, params)
                    else:
                        params = None  # Extend as needed

            if isinstance(country_df, pd.Series):
                country_df = country_df.to_frame()

            if model_name in ['sarimax', 'varmax', 'var_model']:
                predictors = country_df.columns[1:]
            else:
                predictors = None

            config = {
                'model': model,
                'params': params,
                'data': country_df,
                'country': key,
                'indicator': 'Inflation',
                'target': country_df.columns[0],
                "predictors": predictors,
                "log_transform": log_transform,
                "walkforward": walkforward,
                "pt": pt,
            }

            forecast = get_forecast(config)

            del config, country_df
            gc.collect()

            print(f"Completed processing {key} with model {model_name}")

            return forecast
        except Exception as e:
            print(f"Error processing {key}: {e}")
            #print all info about error 
            import traceback
            traceback.print_exc()
            return None
        

def run_monthly_forecast(model, data, countries, log_transform=False, walkforward=False):
    if not countries:
        countries = list(data.keys())

    model_name = model.__name__

    PARAMS_PATH = r"C:\Users\tfarotimi\Documents\CEPAL\Inflation Forecasting\dat\forecast_params.yaml"
    with open(PARAMS_PATH, 'r') as file:
        parameters = yaml.load(file, Loader=yaml.FullLoader)

    print("Running forecasts in parallel...")

    # Run forecasts in parallel
    # with parallel_backend("loky", inner_max_num_threads=1):
    #     forecast_list = Parallel(n_jobs=-1)(
    #         delayed(process_country)(key, data[key], model, model_name, parameters, log_transform, walkforward)
    #         for key in tqdm(data.keys()) if key in countries
    #     )

    forecast_list = []

    for key in tqdm(data.keys()):
        if key not in countries:
            continue
        else:
            forecast = process_country(key, data, model, model_name, parameters, log_transform, walkforward)
            forecast_list.append(forecast)

    # Remove None results (in case of errors)
    forecast_list = [f for f in forecast_list if f is not None]

    return forecast_list


    