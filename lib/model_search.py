
import pandas as pd
import pickle

from sympy import rf
from monthly_forecast_run import run_monthly_forecast
from models import arima, sarimax, varmax, random_forest_w_lags


def run_models (data):
    """
    function to search for best model for each country in data

    Args:
        data (dict): dictionary of dataframes with country name as key and dataframe as value

    Returns:
    dict: dictionary of forecast results with country name as key and dictionary of results as value

    """

    #add _orig to keys in data
    data_orig = {}
    data_lt = {}
    exog_data = {}
    exog_data_lt = {}
    for key in data.keys():
     
        data_orig[key + '_raw'] = data[key].copy()
        data_lt[key + '_lt'] = data[key].copy()
        exog_data[key+'_raw'] = data[key].copy().dropna()
        exog_data_lt[key + '_lt'] = data[key].copy().dropna()


    arima_results = run_monthly_forecast(arima, data_orig, data_orig.keys(), log_transform=False)
    sarimax_results = run_monthly_forecast(sarimax, exog_data, exog_data.keys(), log_transform=False)
    varmax_results = run_monthly_forecast(varmax, exog_data, exog_data.keys(), log_transform=False)

    rf_results = run_monthly_forecast(random_forest_w_lags, data_orig, data_orig.keys(), log_transform=False) 

    # lt_arima_results = run_monthly_forecast(arima, data_lt, data_lt.keys(), log_transform=True)

    # lt_sarimax_results = run_monthly_forecast(sarimax, exog_data_lt, data_lt.keys(), log_transform=True)

    

    data_lt = {}
    for key in data.keys():
        data_lt[key + '_lt'] = data[key].copy()
        
    lt_varmax_results = run_monthly_forecast(varmax, exog_data_lt, exog_data_lt.keys(), log_transform=True)

    model_results = {'arima_results':arima_results, 
                     'sarimax_results':sarimax_results,
                     'varmax_results':varmax_results,
                    #  'lt_arima_results':lt_arima_results,
                    #  'lt_sarimax_results':lt_sarimax_results,
                    #  'lt_varmax_results':lt_varmax_results
                     'rf_results':rf_results
                     }

    #store results for each country and model in object
    results = {}
    for mod, res in model_results.items():
        for c_res in res:
            country = c_res['country'].split('_')[0]
            result = c_res
            results.update({country + mod : result})





    results_df = pd.DataFrame(columns = ['Country', 'Model', 'Params', 'Log Transform', 'RMSE', 'Naive RMSE', 'MAPE', 'Naive MAPE'])
    best_results = []
    for country in data.keys():
        for model in results.keys():
            if country[0:3] in model:
                c = country
                m = results[model]['model']
                p = results[model]['params']
                lt = results[model]['log_transform']
                rmse = float(results[model]['test_metrics']['RMSE'])
                naive_rmse = results[model]['test_metrics']['Naive RMSE']
                mape = results[model]['test_metrics']['MAPE']
                naive_mape = results[model]['test_metrics']['Naive MAPE']
                results_set = model

            
                results_df = pd.concat([results_df, pd.DataFrame([[c, m, p, lt, rmse, naive_rmse, mape, naive_mape, results_set]], columns = ['Country', 'Model', 'Params', 'Log Transform', 'RMSE', 'Naive RMSE', 'MAPE', 'Naive MAPE', 'Results Set'])])


        best_model = results_df[results_df['Country'] == country].sort_values(by = ['RMSE'], ascending = True).drop_duplicates(subset = ['Country'], keep = 'first').reset_index(drop = True)['Results Set'].iloc[0]
        print("Best Model for ", country, " is ", best_model)
        best_results.append(results[best_model])




    results_df = results_df.sort_values(by = ['Country', 'RMSE'], ascending = True).reset_index(drop = True) 
    print(results_df)
    #get lowest rmse for each country
   
    results_obj = {'all': results_df, 'best': best_results}
    timestamp = pd.Timestamp.now()
    filename = 'best_results_obj_' + str(timestamp.hour) + str(timestamp.minute)
    all_results_filename = 'all_results_obj_' + str(timestamp.hour) + str(timestamp.minute)

    '''
    #pickle results_obj
    with open(f'{filename}.pkl', 'wb') as f:
        pickle.dump(results_obj, f)

    with open(f'{all_results_filename}.pkl', 'wb') as f:
        pickle.dump(results, f)

    '''
    
    return results_obj