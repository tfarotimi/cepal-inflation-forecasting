# This file contains the functions for building different forecast models.
# Last update: 08.14.2023 by Inflation Forecasting Farotimi

#import libraries
import array
from ast import mod
from re import M
import re
import sys

from sympy import im

sys.path.append(r"C:\Users\tfarotimi\Documents\CEPAL\Inflation Forecasting\lib")

import pandas as pd
import numpy as np
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.api import VAR
from statsmodels.tsa.statespace.varmax import VARMAX
from statsmodels.tsa.arima.model import ARIMA
from helper_functions import prepare_data, step_forecast, format_results, time_delay_embedding, adfuller_test, invert_transformation
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
#predefined split for cross validation
from sklearn.model_selection import PredefinedSplit
from sklearn.linear_model import LinearRegression
from IPython import embed

import pdb


import shap
import gc

from tqdm.auto import tqdm
from tqdm_joblib import tqdm_joblib
from joblib import parallel_backend



import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import time

#import convolutional neural network
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Flatten, Dropout, LSTM, BatchNormalization, GlobalMaxPooling1D
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit, HalvingRandomSearchCV, cross_val_predict
import numpy as np

from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, make_scorer
from scipy.special import boxcox, inv_boxcox
pio.renderers.default = "browser"
#pt
from sklearn.preprocessing import PowerTransformer
#MATPLOTLIB
import matplotlib.pyplot as plt


from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.stattools import adfuller

from sklearn.preprocessing import MinMaxScaler





import pmdarima as pmd
import yaml




#build mean forecast model
def mean_forecast(data, country, target, predictors = None):
    '''
    Build mean forecast model.

    Parameters 
    ----------
    data : pandas dataframe
        Dataframe containing the data to be used for the forecast model.
    country : str
        Country of interest.
    target : str
        Target variable of interest.
    predictors : list
        List of predictor variables to be used in the model other than target
        variable.

    Returns
    -------
    results : dict
        Dictionary containing the results of the forecast model.


    '''
    model_name = 'Mean Forecast'

    #prepare data
    X_tr, Y_tr, X_ts, Y_ts = prepare_data(data, target, predictors)

    predictions = pd.DataFrame(index = Y_ts.index)
    actuals = pd.DataFrame(Y_ts)

    #create predictions dataframe with mean of last 12 months
    predictions[0] = Y_ts.mean()

    #format results
    results = format_results(country, model_name, target, predictions, actuals)

    #add meta data to results
    results.meta['train_start'] = "N/A"
    results.meta['train_end'] = "N/A"
    results.meta['lags'] = "N/A"
    results.meta['bic'] = "N/A"
    results.meta['model_name'] = model_name

    return results



#build naive forecast model
def naive_forecast(data, country, target, predictors = None, model_name = None, log_transform = False, pt = None):
    '''
    Build naive forecast model where the forecast is the same as the previous month. This serves as a baseline model for comparison.

    Parameters
    ----------
    data : pandas dataframe
        Dataframe containing the data to be used for the forecast model.
    country : str
        Country of interest.    
    target : str
        Target variable of interest.
    predictors : list
        List of predictor variables to be used in the model other than target variable.

    Returns
    -------
    results : dict
        Dictionary containing the results of the forecast model.

    '''


    if model_name != 'Random Forest w/ Lags':
        #create predictions dataframe with 12 months where each month is the value of the last month in the training set
            #prepare data
        X_tr, Y_tr, X_ts, Y_ts = prepare_data(data, target, predictors)
        actuals = pd.DataFrame(Y_ts)

        # Naive forecast: predict the last value of Y_tr for all test points
        last_value = Y_tr.iloc[-1]
        predictions = pd.DataFrame(last_value, index=Y_ts.index, columns=[Y_ts.name if hasattr(Y_ts, 'name') else target])
    else:
        # print('data')
        Y_ts = data['y-test']
        Y_tr = data['y-train']
        X_tr = data['X-train']	
        actuals = pd.DataFrame(Y_ts)
                        # Create a naive forecast: repeat last actual value for all horizons
        last_values = Y_tr.iloc[-1]  # shape: (12,)
        
        naive_forecast = np.tile(last_values, (Y_ts.shape[0], 1))  # shape: (12, 12)
        # print("naive_forecast", naive_forecast)
        predictions = pd.DataFrame(naive_forecast, index=Y_ts.index, columns=Y_ts.columns)
        # print("Naive forecast model created with last values from training set.")


    if log_transform:
        #transform data
        #make pred arr the same as the last value in Y_tr repeated 12 times
        pred_arr =np.tile(np.array(Y_tr.iloc[-1]).reshape(-1,1), (len(Y_ts), 1))

        pred_arr = pt.inverse_transform(pred_arr).reshape(-1,1).squeeze()
        # pred_arr = (20 * np.exp(pred_arr) +1)  / (1 + np.exp(pred_arr))

        
        #pred_arr = pt.inverse_transform(np.array(Y_ts.shift(1)[1:]).reshape(-1,1)).squeeze()
        predictions = pd.DataFrame(pred_arr, index = Y_ts.index)

        actual_arr = pt.inverse_transform(np.array(Y_ts).reshape(-1,1)).squeeze()
        # actual_arr = (20 * np.exp(np.array(Y_ts)) +1)  / (1 + np.exp(np.array(Y_ts)))
        actuals = pd.DataFrame(actual_arr, index = Y_ts.index)




    #format results
    results = format_results(country, model_name, target, predictions, actuals)

    results.meta['train_start'] = "N/A"
    results.meta['train_end'] = "N/A"
    results.meta['lags'] = "N/A"
    results.meta['bic'] = "N/A"
    results.meta['mape'] = round(100 * np.mean(np.abs(np.array(predictions) - np.array(actuals))/np.array(actuals)),2)
    results.meta['model_name'] = model_name if model_name else "Naive Forecast"
    return results


#build univariate ARDL model
def univar_ARDL(data, country, target, lags, predictors = None, walk_forward = False):

    '''
    Build Univariate AutoRegressive Distrubted Lags model.

    Parameters
    ----------
    data : pandas dataframe
        Dataframe containing the data to be used for the forecast model.
    country : str
        Country of interest.    
    target : str
        Target variable of interest.
    predictors : list
        List of predictor variables to be used in the model other than target variable.

    Returns
    -------
    results : dict
        Dictionary containing the results of the forecast model.

    '''
    model_name = 'Univariate ARDL'

    #prepare data
    X_tr, Y_tr, X_ts, Y_ts = prepare_data(data, target, predictors)

    if (walk_forward):
        model_name = model_name
        predictions = list()
    
        tqdm._instances.clear()
        for i in tqdm(range(len(X_ts)), desc="Testing "+model_name, position=0, leave=True):
          
            next_X = X_ts.iloc[[i]]
            next_y = Y_ts.iloc[[i]]
            # fit model on history and make a predictionF
            model_fit, pred = step_forecast("univar_ARDL", X_tr, Y_tr, next_X,counter = i, n_steps = len(X_ts))
            # store forecast in list of predictions
            predictions.append(pred)
            # add actual observations to history for the next loop
            X_tr = pd.concat([X_tr, next_X])
            Y_tr = pd.concat([Y_tr, next_y])
        
        predictions = pd.DataFrame(predictions)
        
    else:
        #fit model
        model = AutoReg(Y_tr, lags=12)
        model_fit = model.fit()
        

    

        predictions = pd.DataFrame(model_fit.predict(start=len(Y_tr), end=len(Y_tr)+len(Y_ts)-1, dynamic= False))

    #format results
    results = format_results(country, model_name, target, predictions, Y_ts)


    #add train start and end dates to meta
    results.meta['train_start'] = data.index[0].strftime("%Y-%m-%d")
    results.meta['train_end'] = data.index[len(Y_tr)-1].strftime("%Y-%m-%d")
    results.meta['lags'] = lags
    results.meta['model'] = model_fit
    results.meta['bic'] = round(model_fit.bic,2)
    results.meta['model_name'] = model_name


    #create new attribute for results called train
    results.train = {}
    results.train['X_tr'] = X_tr
    results.train['X_ts'] = X_ts



    return results

    

#build multivariate ARDL model
def multivar_ARDL(data, country, target, predictors = None, walk_forward = False):
    '''
    Build multivariate AutoRegressive Distribution Lags forecast model.

    Parameters
    ----------
    data : pandas dataframe
        Dataframe containing the data to be used for the forecast model.
    country : str
        Country of interest.    
    target : str
        Target variable of interest.
    predictors : list
        List of predictor variables to be used in the model other than target variable.

    Returns
    -------
    results : dict
        Dictionary containing the results of the forecast model.

    '''

    model_name = 'Multivariate ARDL'
    
    #prepare data
    X_tr, Y_tr, X_ts, Y_ts = prepare_data(data, target, predictors)

    
    

    if (walk_forward):
        model_name = model_name + " - Walk Forward"

        predictions = list()

        tqdm._instances.clear()
        #run model on testing data
        for i in tqdm(range(len(X_ts)), desc="Testing "+model_name, position=0, leave=True):
          
            next_X = X_ts.iloc[[i]]
            next_y = Y_ts.iloc[[i]]
            # fit model on history and make a prediction
            model_fit, pred = step_forecast("multivar_ARDL", X_tr, Y_tr, next_X,counter = i, n_steps = len(X_ts))
            # store forecast in list of predictions
            predictions.append(pred)
            # add actual observations to history for the next loop
            X_tr = pd.concat([X_tr, next_X])
            Y_tr = pd.concat([Y_tr, next_y])

        predictions = pd.DataFrame(predictions)

    else:
        #fit model
        model = AutoReg(Y_tr, lags=12, exog=X_tr)
        model_fit = model.fit()

        #make predictions
    
        predictions = pd.DataFrame(model_fit.predict(dynamic=False, exog=X_tr,exog_oos=X_ts,start=len(Y_tr), end=len(Y_tr)+11))
        
    #format results
    results = format_results(country, model_name, target, predictions, Y_ts)


    #add meta data to results
    results.meta['train_start'] = data.index[0].strftime("%Y-%m-%d")
    results.meta['train_end'] = data.index[len(Y_tr)-1].strftime("%Y-%m-%d")
    results.meta['lags'] = 12
    results.meta['features'] = X_tr.columns.tolist()
    results.meta['model'] = model_fit
    results.meta['bic'] = round(model_fit.bic,2)
    results.meta['model_name'] = model_name

    #create new attribute for results called train
    results.train = {}
    results.train['X_tr'] = X_tr
    results.train['X_ts'] = X_ts


    
    return results

#build random forest model
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Assume time_delay_embedding and format_results are defined elsewhere


def random_forest_w_lags(data, country, target, lags=12, horizon=0, walk_forward=False, predictors=None):
    """
    Build a Random Forest forecast model using lagged inflation data.

    Parameters
    ----------
    data : pandas DataFrame
        Input data containing the target and optional predictors.
    country : str
        Country name (for results annotation).
    target : str
        Column name of the target variable.
    lags : int
        Number of lagged values to include.
    horizon : int
        Forecast horizon (months ahead).
    walk_forward : bool
        Whether to use walk-forward validation.
    predictors : list
        List of additional exogenous predictors (not used if None).

    Returns
    -------
    results : dict or custom results object
        Results containing forecasts, actuals, metadata, and model.
    """
    model_name = "Random Forest w/ Lags" if lags else "Random Forest w/o Lags"
    results = pd.DataFrame()

    max_lag      = 12       # current + max_lag back -> max_lag+1 lag features
    horizon      = 0       # how many steps ahead you want to predict
    test_size    = 12       # number of final rows in the test set
    roll_windows = [3, 6]  

    # Select target and predictors
    if predictors:
        features = predictors + [target]
        data = data[features]
    else:
        data = data[0][[target]] if isinstance(data, tuple) else data[[target]]

    #Create lagged features
    lagged_data = []
    for col in data:
        col_df = time_delay_embedding(data[col], n_lags=lags, horizon=horizon)
        lagged_data.append(col_df)

    data = pd.concat(lagged_data, axis=1).dropna()


    #merge with data from "exog_lagged_data.xlsx" if it exists
    try:
        # Load exogenous data
        exog_data = pd.read_excel(
            r"C:\Users\tfarotimi\Documents\CEPAL\Inflation Forecasting\dat\exog_lagged_data.xlsx",
            index_col=0
        )

        # Normalize and align index
        exog_data.index = pd.to_datetime(exog_data.index).to_period('M').to_timestamp()
        exog_data.index.name = 'Period'

        # Do the same to main data
        data.index = pd.to_datetime(data.index).to_period('M').to_timestamp()
        data.index.name = 'Period'

        # Diagnostic: Check intersection
        overlap = data.index.intersection(exog_data.index)
        print(f"✅ Overlap: {len(overlap)} timestamps match")

        # Optional: trim to shared range
        start = max(data.index.min(), exog_data.index.min())
        end = min(data.index.max(), exog_data.index.max())
        data = data.loc[(data.index >= start) & (data.index <= end)]

        # Merge
        data = data.merge(exog_data, how='left', left_index=True, right_index=True)

        # Final sanity check
        print("✅ Merge successful. Shape:", data.shape)
    except FileNotFoundError:
        print("No exogenous lagged data found. Proceeding with available data.")
    # Ensure the index is a datetime index
    print("Training data after featuring engineering: ", data.shape)

    #stop execution if dataframes contain NaN values
    if data.isnull().values.any():
        raise ValueError("Data contains NaN values after feature engineering. Please check your data.")
    

    # Split into predictors and target
    target_mask = data.columns.str.contains(f'\\(t\\-')
    X = data.loc[:, target_mask]
    Y = data.loc[:, ~target_mask]

    X["trend"] = np.arange(len(data))                       # linear time index
    X["month"] = data.index.month.astype("category")        # seasonality dummies
    X = pd.get_dummies(X, columns=["month"], drop_first=True)

    # 4)  rolling statistics (mean & std) ---------------------------------------
    for w in roll_windows:
        X[f"roll{w}_mean"] = Y['YoY Increase Inflation(t)'].rolling(w).mean()
        X[f"roll{w}_std"]  = Y['YoY Increase Inflation(t)'].rolling(w).std()

    # 5)  YoY and MoM percentage changes ----------------------------------------
    X["mom"] = Y['YoY Increase Inflation(t)'].pct_change()                # month-on-month
    X["yoy"] = Y['YoY Increase Inflation(t)'].pct_change(12)              # year-on-year

    X = X.dropna()  # drop rows with NaN values after feature engineering
    Y = Y.loc[X.index]  # align Y with X after dropping NaNs

    print("X_shape:", X.shape, "Y shape:", Y.shape)
    
    # Train/test split
    X_tr, X_ts = X[:-12], X[-12:]
    Y_tr, Y_ts = Y[:-12], Y[-12:]

    #sort X_tr and X_ts columns 
    X_tr = X_tr.reindex(sorted(X_tr.columns), axis=1)
    X_ts = X_ts.reindex(sorted(X_ts.columns), axis=1)


    #write X_tr, Y_tr, X_ts, Y_ts to excel file 

    # print("last X_tr", X_tr.iloc[-1])
    # print("last Y_tr", Y_tr.iloc[-1])
    # print("last X_ts", X_ts.iloc[-1])
    # print("last Y_ts", Y_ts.iloc[-1])

    #add to 

    if walk_forward:
        model_name += " - Walk Forward"
        preds = []
        tqdm._instances.clear()
        for i in tqdm(range(len(X_ts)), desc="Testing "+model_name, position=0, leave=True):
            next_X = X_ts.iloc[[i]]
            next_y = Y_ts.iloc[[i]]
            model = RandomForestRegressor()
            model.fit(X_tr, Y_tr)
            pred = model.predict(next_X)
            preds.append(pred)
            X_tr = pd.concat([X_tr, next_X])
            Y_tr = pd.concat([Y_tr, next_y])
    else:
        param_dist = {
            "n_estimators": [50,75,100,125,150],
            "max_depth": [None] + list(np.arange(200, 1100, 100)),
            "max_features": ["sqrt", "log2", 0.5, 0.7],
            "min_samples_split": [2, 5, 10, 15],
            "min_samples_leaf": [1, 2, 4, 8],
            "bootstrap": [True, False]
        }

        rf   = RandomForestRegressor(random_state=42, n_jobs=1)   # ← 1 core **inside** each fit

       #randomized search 
        rmse_scorer = make_scorer(mean_squared_error, greater_is_better=False, squared=False)

        n_iter = 75  # Number of iterations for RandomizedSearchCV
        tscv = TimeSeriesSplit(n_splits=5, test_size=12)  # Time series cross-validation
       
        n_splits = tscv.get_n_splits()
        total_fits = n_splits * n_iter  # Total number of fits to be performed

        #tqdm
        tqdm._instances.clear()  # Clear any previous tqdm instances

        # ← all outer cores
        search_multi = RandomizedSearchCV(
            estimator=rf,
            param_distributions=param_dist,
            n_iter=75,
            cv=tscv,
            scoring=rmse_scorer,
            n_jobs=-1,
            verbose=0,
            refit=True,
            random_state=42
        )



        print("Training Random Forest model with RandomizedSearchCV...")
        start_time = time.time()
        with tqdm_joblib(tqdm(desc="Random search fits", total=total_fits)):
            with parallel_backend("loky", inner_max_num_threads=1):
                search_multi.fit(X_tr, Y_tr)
        elapsed_time = time.time() - start_time
        print(f"Random Forest model training time: {elapsed_time:.2f} seconds")

       # Finalize model after RandomizedSearchCV

        print("Best parameters found:", search_multi.best_params_)
        print("Best score (RMSE):", -search_multi.best_score_)
        print("Training updated model with best parameters...for intermediate steps")
        start_time = time.time()
        model = search_multi.best_estimator_.set_params(
            n_estimators=800,
            max_depth=None,
            n_jobs=-1
        )
        model.fit(X_tr, Y_tr)
        elapsed_time = time.time() - start_time
        print(f"Best Random Forest model training time: {elapsed_time:.2f} seconds")

        # Build target series from all available data
        target = "YoY Increase Inflation(t)"
        full_target_series = pd.Series(Y_tr[target])


        best_params = search_multi.best_params_

        del search_multi  # free memory after search
        # clear garbage collector
        gc.collect()

        # ── 1. non-overlapping OOF folds ──────────────────────────────
        # ── 1. non-overlapping OOF folds ──────────────────────────────
        # Purpose: Compute out-of-fold (OOF) predictions on training data.
        # This helps estimate model generalization error *during training* without leaking test information.
        # Useful for bias analysis, residual diagnostics, and comparison with in-sample performance.
        # Each sample is predicted by a model that has not seen it during training (like cross-validation).

        n_samples = len(X_tr)
        fold_size = n_samples // 4
        oof_pred = np.full_like(Y_tr, np.nan, dtype=float)

        for fold_id in range(1, 5):  # 4 folds
            train_end = fold_id * fold_size
            test_start = train_end
            test_end = min(test_start + fold_size, n_samples)

            if test_start >= n_samples:
                break

            train_idx = np.arange(0, train_end)
            test_idx = np.arange(test_start, test_end)
            
            print(f"Fold {fold_id}: train={len(train_idx)}, test={len(test_idx)}")

            model_cv = RandomForestRegressor(
                **best_params,
                n_jobs=1,
                random_state=fold_id
            ).fit(X_tr.iloc[train_idx], Y_tr.iloc[train_idx])

            oof_pred[test_idx] = model_cv.predict(X_tr.iloc[test_idx]).reshape(-1, oof_pred.shape[1])

        # Create DataFrame
        oof_df = pd.DataFrame(oof_pred, index=X_tr.index, columns=Y_tr.columns)

        # ✅ Only compute RMSE where predictions exist
        valid_mask = ~np.isnan(oof_pred).any(axis=1)
        oof_rmse = mean_squared_error(Y_tr[valid_mask], oof_df[valid_mask], squared=False)


        plt.figure(figsize=(12, 5))
        plt.plot(Y_tr.index, Y_tr.values, label='Actual', color='black')
        plt.plot(Y_tr.index, oof_df.values, label='OOF Prediction', color='red')
        plt.title("Out-of-Fold Predictions vs. Actual")
        plt.legend()
        plt.grid(True)
        plt.show()

        print(f"Out-of-fold RMSE: {oof_rmse:.4f}")


        residuals = (Y_tr.values.ravel() - oof_df.values.ravel())

        plt.hist(residuals, bins=30, color="skyblue", edgecolor="k")
        plt.title("Out-of-Fold Residuals")
        plt.xlabel("Residual")
        plt.ylabel("Count")
        plt.grid(True)
        plt.show()

        #remove n_estimators from best_params
        best_params.pop("n_estimators", None)  # remove n_estimators from best_params
        #remove oob_score from best_params
        best_params.pop("oob_score", None)  # remove oob_score from best_params
        #remove bootstrap from best_params
        best_params.pop("bootstrap", None)  # remove bootstrap from best_params

        


            # ──────────────────────────────────────────────────────────────
        # 1.  Sweep tree counts with out-of-bag RMSE to find optimal tree count
        # ──────────────────────────────────────────────────────────────
        # Sweep settings
        tree_grid   = [150, 300, 500, 800, 1000, 1200, 1400, 1500]
        oob_curve   = {}

        print("Testing different tree counts with tqdm progress bar...")

        early_stop_threshold = 0.001
        patience = 2
        no_improve_count = 0
        prev_rmse = None

        tscv = TimeSeriesSplit(n_splits=5)

        with tqdm(total=len(tree_grid), desc="Tree count sweep") as pbar:
            for n in tree_grid:
                rmses = []

                for train_index, val_index in tscv.split(X_tr):
                    X_train, X_val = X_tr.iloc[train_index], X_tr.iloc[val_index]
                    Y_train, Y_val = Y_tr.iloc[train_index], Y_tr.iloc[val_index]

                    rf_oob = RandomForestRegressor(
                        **best_params,
                        n_estimators=n,
                        bootstrap=True,
                        n_jobs=-1,
                        random_state=42
                    ).fit(X_train, Y_train)

                    preds = rf_oob.predict(X_val)
                    rmse = mean_squared_error(Y_val.values.ravel(), preds.ravel(), squared=False)
                    rmses.append(rmse)

                    del rf_oob  # free memory
                    gc.collect()

                avg_rmse = np.mean(rmses)
                oob_curve[n] = avg_rmse
                print(f"{n} trees → avg CV RMSE = {avg_rmse:.3f}")
                pbar.update(1)

                # Early stopping
                if prev_rmse is not None:
                    delta = abs(prev_rmse - avg_rmse)
                    print(f"Δ RMSE from previous: {delta:.5f}")
                    if delta < early_stop_threshold:
                        no_improve_count += 1
                        print(f"No improvement count: {no_improve_count}")
                        if no_improve_count >= patience:
                            print(f"Early stopping triggered at {n} trees.")
                            break
                    else:
                        no_improve_count = 0
                prev_rmse = avg_rmse

        # Best result
        best_n = min(oob_curve, key=oob_curve.get)
        avg_cv_rmse = oob_curve[best_n]
        print(f"\n✅ Best number of trees: {best_n} with avg CV RMSE = {avg_cv_rmse:.3f}")

        # Plot
        plt.figure(figsize=(8, 5))
        plt.plot(list(oob_curve.keys()), list(oob_curve.values()), marker="o", linestyle="-", color="blue")
        plt.title("CV RMSE vs. Number of Trees", fontsize=14)
        plt.xlabel("Number of Trees")
        plt.ylabel("Cross-Validated RMSE")
        plt.grid(True)
        plt.tight_layout()
        plt.show()


       
        # ──────────────────────────────────────────────────────────────
        # 2.  Final refit with the chosen tree count
        # ──────────────────────────────────────────────────────────────
        final_params = best_params | {"n_estimators": best_n, "n_jobs": -1}
       # Train final model (core_model)
        core_model = RandomForestRegressor(**final_params, random_state=42)
        print("Training final Random Forest model with best parameters...")
        start_time = time.time()
        core_model.fit(X_tr, Y_tr)
        elapsed_time = time.time() - start_time
        print(f"Final model training time: {elapsed_time:.2f} seconds")

        # Use legacy TreeExplainer for tree-based models to avoid GPU issues
        explainer = shap.TreeExplainer(core_model)
        shap_values = explainer.shap_values(X_tr)

        # Plot global feature importance as bar chart
        shap.summary_plot(shap_values, X_tr, plot_type="bar")

        #store shap values in results 
        
        # clear garbage collector

        # Optional: Show SHAP beeswarm plot
        shap.summary_plot(shap_values, X_tr)
        # del shap_values  # free memory after plotting
        # gc.collect()  # clear memory after plotting


        # ──────────────────────────────────────────────────────────────
        # 3.  (Optional) recompute bias — will usually be ~0
        # ──────────────────────────────────────────────────────────────
        # Calculate bias on training residuals — for diagnostics only
        train_bias = (Y_tr.to_numpy().ravel() - core_model.predict(X_tr)).mean()
        print(f"Training set bias (mean residual): {train_bias:.4f}")


        # Then run recursive forecast
        print("Running recursive forecast with feature generation...")
        start_time = time.time()
        #recursive_forecast = recursive_forecast_rf_strict(core_model, full_series=full_target_series)
        forecast = core_model.predict(X_ts)
        print("Forecast", forecast)
        elapsed_time = time.time() - start_time
        print(f"Recursive forecast time: {elapsed_time:.2f} seconds")

        # Store final prediction as DataFrame (for compatibility with Y_ts)
        preds = pd.DataFrame(forecast, index=Y_ts.index, columns=Y_ts.columns)

                # Apply correction to the strict recursive forecast (optional)
        y_hat_ts = preds.values

        # RMSE against true test values
        bi_co_residuals = Y_ts - y_hat_ts




        # Convert predictions to DataFrame
        rmse_ts = mean_squared_error(Y_ts.to_numpy(), y_hat_ts, squared=False)
        print("rmse_ts:", rmse_ts)
        print("y hat ts:", y_hat_ts)


        #print("raw pred:", raw_pred)
        # corrected  = prod_model.predict(X_ts)

        # print("Mean residual before corr :", (Y_ts - raw_pred).mean(axis=0)[0])
        # print("Mean residual after  corr :", (Y_ts - corrected).mean(axis=0)[0])

        # Prepare predictions and actuals as DataFrames for format_results
        predictions = preds.copy()
        y_actual = pd.DataFrame(Y_ts, index=Y_ts.index, columns=Y_ts.columns)

        print("Formatting results...")
        results = format_results(country, model_name, target, predictions, y_actual)
        print("fine after formatting results")

        # Add metadata
        results.meta['train_start'] = data.index[0].strftime("%Y-%m-%d")
        results.meta['train_end'] = data.index[len(Y_tr)-1].strftime("%Y-%m-%d")
        results.meta['lags'] = lags
        results.meta['features'] = X_tr.columns.tolist()
        results.meta['model'] = core_model
        results.meta['feature_names'] = core_model.feature_names_in_.tolist() if hasattr(core_model, 'feature_names_in_') else "N/A"
        results.meta['feature_importances'] = core_model.feature_importances_ if hasattr(core_model, 'feature_importances_') else "N/A"
        results.meta['bic'] = 3  # Placeholder
        results.meta['X_train'] = X_tr
        results.meta['X_test'] = X_ts
        results.meta['y_train'] = Y_tr
        results.meta['y_test'] = Y_ts
        results.meta['model_name'] = model_name
        results.meta['bias_vec'] = train_bias if model_name == "Random Forest w/ Lags" else "N/A"
        results.meta['prod_model'] = core_model if model_name == "Random Forest w/ Lags" else "N/A"
        results.meta['y_hat_ts'] = y_hat_ts if model_name == "Random Forest w/ Lags" else "N/A"
        results.meta['rmse_ts'] = rmse_ts if model_name == "Random Forest w/ Lags" else "N/A"
        results.meta['bi_co_residuals'] = bi_co_residuals if model_name == "Random Forest w/ Lags" else "N/A"
        results.meta['params'] = final_params if model_name == "Random Forest w/ Lags" else "N/A"
        results.meta['shap_values'] = shap_values


    
        return results


def sarimax(data, country, target, params,predictors = None, walkforward=False, log_transform = False, pt = None):
    '''
    Build SARIMAX model.

    Parameters
    ----------
    data : pandas dataframe
        Dataframe containing the data to be used for the forecast model.
    country : str
        Country of interest.    
    target : str
        Target variable of interest.
    predictors : list
        List of predictor variables to be used in the model other than target variable.

    Returns
    -------
    results : dict
        Dictionary containing the results of the forecast model.

    '''
    model_name = "SARIMAX"
    X_tr, Y_tr, X_ts, Y_ts = prepare_data(data, target, predictors)

  
    o = params[0]
    s_o  = params[1]
    model = SARIMAX(Y_tr, exog=X_tr, order=o,
                            seasonal_order=s_o,
                            enforce_stationarity=False,
                            enforce_invertibility=False,
                            innovations = 't'
                            )

    model_fit = model.fit()

    #forecast exog for use in validation
    future_exog = pd.DataFrame()

    with open(r'C:\Users\tfarotimi\Documents\CEPAL\Inflation Forecasting\dat/predictor_params.yaml', 'r') as file:
        parameters = yaml.load(file, Loader=yaml.FullLoader)

    #check if country exists in yaml file, if not create new entry
    if country not in parameters:
        parameters[country] = {}

    
    for i in predictors:
        #special case for YoY variables, if exists in 'ARG_raw', use params from yaml file to fit arima (change to global later)
        if 'YoY' in i:
            country = 'ARG_raw'
        #check if i exists in predictors_params.yaml
        if i in parameters[country]:
            #if it exists, use params from yaml file to fit arima
            params = parameters[country][i]['params']
            exog_model = ARIMA(X_tr[i], order=params['order'], seasonal_order=params['seasonal_order'], enforce_stationarity=False, enforce_invertibility=False).fit()
            future_exog[i] = exog_model.forecast(12)
        else:
            #run auto arima and predict 12 months of X[i] with best params
            best_arima = pmd.auto_arima(X_tr[i], start_p=0, start_q=0,
                                    test= 'kpss',   # use kpss to find optimal 'd'
                                    max_p=3,  max_q=3,
                                    start_P = 0, start_Q=0,
                                    max_P = 3,  max_Q = 3,
                                    seasonal = True,
                                    trace=False,
                                    m=12,             
                                    stepwise=True,
                                    scoring='mse')
        
            exog_model = best_arima.fit(X_tr[i])

            #add params to yaml file for the country if it exists, else create new entry
            if country in parameters:
                parameters[country][i] = {'params': best_arima.get_params()}
            else:
                parameters[country] = {i: {'params': best_arima.get_params()}}

            #write to yaml file
            with open(r'C:\Users\tfarotimi\Documents\CEPAL\Inflation Forecasting\dat/predictor_params.yaml', 'w') as file:
                yaml.dump(parameters, file)
        
            future_exog[i] = exog_model.predict(n_periods=12)

        
        #end of forecast exog

    preds = model_fit.forecast(steps=12, exog = future_exog)

    if log_transform:
           #transform data
        pred_arr = pt.inverse_transform(np.array(preds).reshape(-1,1)).squeeze()
        predictions = pd.DataFrame(pred_arr, index = preds.index)

        actual_arr = pt.inverse_transform(np.array(Y_ts).reshape(-1,1)).squeeze()
        y_actual = pd.DataFrame(actual_arr, index = Y_ts.index)

    else:
        predictions = pd.DataFrame(preds)
        y_actual = pd.DataFrame(Y_ts)

    #calculate mase with in-sample naive forecast
    # insample_mae_list = []
  
    # for i in range(1,len(Y_tr)-12):
    #     insample_mae_list.append(np.mean(np.abs(np.array(Y_tr[i:i+12]) - Y_tr[i-1]*np.ones(12)))) #in-sample naive forecast is the last value in the training set, error is the difference between the last value and the actual value for the next 12 months

    # insample_mae = np.mean(insample_mae_list) #average naive forecast error

    #calculate mase with out-of-sample naive forecast
    residuals = np.array(predictions) - np.array(y_actual) #residuals are the difference between the predictions and actuals
    # mase = np.mean(np.abs(residuals/insample_mae)) #divide each forecast error by the average naive forecast error and take the average of the absolute values
    mape = 100 * np.mean(np.abs(residuals/np.array(y_actual))) #divide each forecast error by the actual value and take the average of the absolute values

    #if mase > 1, then the model is worse than the naive forecast because on average, the forecast error is greater than the naive forecast error
    
    #format results
    results = format_results(country, model_name, target, predictions, y_actual)

    results.meta['train_start'] = data.index[0].strftime("%Y-%m-%d")
    results.meta['train_end'] = data.index[len(Y_tr)-1].strftime("%Y-%m-%d")
    results.meta['lags'] = "N/A"
    results.meta['features'] = X_tr.columns.tolist()
    results.meta['model'] = model_fit
    results.meta['aic'] = round(model_fit.aic,2)
    results.meta['train_rmse'] = round(np.sqrt(model_fit.mse),2)
    results.meta['mape'] = round(mape,2)
    results.meta['model_name'] = model_name

    

    #create new attribute for results called train
    results.train = {}
    results.train['X_tr'] = X_tr
    results.train['X_ts'] = X_ts

    return results

def arima(data, country, target, params, predictors = None, walkforward = False, log_transform = False, pt = None):
    '''
    Build ARIMA model.

    Parameters
    ----------
    data : pandas dataframe
        Dataframe containing the data to be used for the forecast model.
    country : str
        Country of interest.    
    target : str
        Target variable of interest.
    predictors : list
        List of predictor variables to be used in the model other than target variable.

    Returns
    -------
    results : dict
        Dictionary containing the results of the forecast model.

    '''
       
    model_name = "ARIMA"
    X_tr, Y_tr, X_ts, Y_ts = prepare_data(data, target)

    if (walkforward):
        ar_params = {'order': params[0], 'seasonal_order': params[1]}

        model_name = model_name + " - Walk Forward"
        preds = list()
        
        tqdm._instances.clear()
        for i in tqdm(range(len(X_ts)), desc="Testing "+ model_name, position=0, leave=True):
            #clear previous tqdm output
            tqdm._instances.clear()
            next_X = X_ts.iloc[[i]]
            next_y = Y_ts.iloc[[i]]
            # fit model on history and make a prediction
            model_fit, pred = step_forecast("arima", X_tr, Y_tr, next_X,counter = i, n_steps = len(X_ts), params = ar_params)
            # store forecast in list of predictions
            preds.append(pred)
            # add actual observations to history for the next loop
            X_tr = pd.concat([X_tr, next_X])
            Y_tr = pd.concat([Y_tr, next_y])
        
    else:
        o = params[0]
        s_o  = params[1]
        model = ARIMA(Y_tr, order=o,
                                seasonal_order=s_o,
                                enforce_stationarity=False,
                                enforce_invertibility=False)

        model_fit = model.fit()

        preds = model_fit.predict(start = len(Y_tr), end = len(Y_tr)+ 11)


    if log_transform:
        #transform data
        y_actual = pd.DataFrame(Y_ts)

        pred_arr = pt.inverse_transform(np.array(preds).reshape(-1,1)).squeeze()
        predictions = pd.DataFrame(pred_arr, index = preds.index)

        actual_arr = pt.inverse_transform(np.array(Y_ts).reshape(-1,1)).squeeze()
        y_actual = pd.DataFrame(actual_arr, index = Y_ts.index)

        #(b-a)*exp(fc$mean)/(1+exp(fc$mean)) + a

        # pred_arr = (20 * np.exp(preds) +1)  / (1 + np.exp(preds))
        # predictions = pd.DataFrame(pred_arr)


        # actual_arr = (20 * np.exp(y_actual) +1)  / (1 + np.exp(y_actual))
        # y_actual = pd.DataFrame(actual_arr, index = Y_ts.index)
    # else:
    #     y_actual = pd.DataFrame(Y_ts)

    else:
        predictions = pd.DataFrame(preds)
        y_actual = pd.DataFrame(Y_ts)

    # #calculate mase with in-sample naive forecast
    # insample_mae_list = []
  
    # for i in range(12, len(Y_tr)-12):
    #     # Use the last 12 values of Y_tr as the naive forecast for the next 12 months
    #     insample_mae_list.append(np.mean(np.abs(np.array(Y_tr[i:i+12]) - np.array(Y_tr[i-12:i]))))  # naive forecast is last 12 values

    # insample_mae = np.mean(insample_mae_list) #average naive forecast error

    #calculate mase with out-of-sample naive forecast
    residuals = np.array(predictions) - np.array(y_actual) #residuals are the difference between the predictions and actuals
    # mase = np.mean(np.abs(residuals/insample_mae)) #divide each forecast error by the average naive forecast error and take the average of the absolute values
    mape = 100 * np.mean(np.abs(residuals/np.array(y_actual))) #divide each forecast error by the actual value and take the average of the absolute values

    #if mase > 1, then the model is worse than the naive forecast because on average, the forecast error is greater than the naive forecast error
    

    #subtract predictinos from actuals to get residuals

    #format results
    results = format_results(country, model_name, target, predictions, y_actual)

    results.meta['train_start'] = data.index[0].strftime("%Y-%m-%d")
    results.meta['train_end'] = data.index[len(Y_tr)-1].strftime("%Y-%m-%d")
    results.meta['lags'] = "N/A"
    results.meta['features'] = X_tr.columns.tolist() if X_tr is not None else "N/A - No Exogenous Variables"
    results.meta['model'] = model_fit
    results.meta['aic'] = round(model_fit.aic,2)
    results.meta['mape'] = round(mape,2)
    results.meta['train_rmse'] = round(np.sqrt(model_fit.mse),2)
    results.meta['model_name'] = model_name
    

    #create new attribute for results called train
    if X_tr is not None:
        results.train = {}
        results.train['X_tr'] = X_tr
        results.train['X_ts'] = X_ts

    return results


def varmax (data, country, target, params, predictors = None, walk_forward = False, log_transform = False, pt = None, walkforward = False):
    '''
    Build VARMAX model.

    Parameters
    ----------
    data : pandas dataframe
        Dataframe containing the data to be used for the forecast model.
    country : str
        Country of interest.    
    target : str
        Target variable of interest.
    predictors : list
        List of predictor variables to be used in the model other than target variable.

    Returns
    -------
    results : dict
        Dictionary containing the results of the forecast model.

    '''
    model_name = "VARMAX"
    X_tr, Y_tr, X_ts, Y_ts = prepare_data(data, target, predictors)
    
    #concanate X_tr to Y_tr
    varmax_Y = pd.concat([Y_tr, X_tr], axis=1)

    #MAKE Y_tr have shape 1Xn
    Y_tr_array = Y_tr.values.reshape(len(Y_tr),1)

    c = "varmax_"+ country

    #check params for country
    with open(r'C:\Users\tfarotimi\Documents\CEPAL\Inflation Forecasting\dat/forecast_params.yaml', 'r') as file:
        parameters = yaml.load(file, Loader=yaml.FullLoader)

    #check if country exists in yaml file, if not create new entry
    if c in parameters:
         p, d, q = parameters[c]['order']
         P, D, Q, m = parameters[c]['seasonal_order']

         params = {'order': (p,d,q), 'seasonal_order': (P,D,Q,m)}

    else:
        #auto arima for Y_tr
        best_arima = pmd.auto_arima(Y_tr_array, start_p=1, start_q=1,
                                    test= 'kpss',   # use kpss to find optimal 'd'
                                    max_p=3,  max_q=3,
                                    start_P = 1, start_Q=1,
                                    max_P = 3,  max_Q = 3,
                                    seasonal = True,
                                    trace=False,
                                    m=12,
                                    stepwise=True,
                                    scoring='mse')
        p, d, q = best_arima.order
        P, D, Q, m = best_arima.seasonal_order

        params = {'order': (p,d,q), 'seasonal_order': (P,D,Q,m)}

        #add params to yaml file for the country if it exists, else create new entry
        parameters[c] = {}
        parameters[c]['order'] = best_arima.order
        parameters[c]['seasonal_order'] = best_arima.seasonal_order

        #write to yaml file
        with open(r"C:\Users\tfarotimi\Documents\CEPAL\Inflation Forecasting\dat\forecast_params.yaml", 'w') as file:
            yaml.dump(parameters, file)

    if (walkforward):
        model_name = model_name + " - Walk Forward"

        predictions = list()

        tqdm._instances.clear()
        #run model on testing data
        for i in tqdm(range(len(X_ts)), desc="Testing "+model_name, position=0, leave=True):
          
            next_X = X_ts.iloc[[i]]
            next_y = Y_ts.iloc[[i]]
            # fit model on history and make a prediction
            model_fit, pred = step_forecast("varmax", X_tr, Y_tr, next_X,counter = i, n_steps = len(X_ts), params = params)
            # store forecast in list of predictions
            predictions.append(pred)
            # add actual observations to history for the next loop
            X_tr = X_tr.append(next_X)
            Y_tr = Y_tr.append(next_y)
        
         
        preds = pd.DataFrame(predictions)

    else:
        if d == 0:
            model = VARMAX(varmax_Y,order=(p,q), seasonal_order=(P,D,Q,12))
        else:
            model = VARMAX(varmax_Y,order=(3,q), seasonal_order=(P,D,Q,12), trend='n')

        model_fit = model.fit(maxiter=100)
        preds = model_fit.get_forecast(steps=16).predicted_mean.iloc[:,0]

    if log_transform:
            #transform data
        pred_arr = pt.inverse_transform(np.array(preds).reshape(-1,1)).squeeze()
        predictions = pd.DataFrame(pred_arr)

        actual_arr = pt.inverse_transform(np.array(Y_ts).reshape(-1,1)).squeeze()
        y_actual = pd.DataFrame(actual_arr, index = Y_ts.index)
    else:
        predictions = preds
        y_actual = pd.DataFrame(Y_ts)


    #calculate mase with out-of-sample naive forecast
    residuals = np.array(predictions) - np.array(y_actual) #residuals are the difference between the predictions and actuals
    mape = 100 * np.mean(np.abs(residuals/np.array(y_actual))) #divide each forecast error by the actual value and take the average of the absolute values

    #if mase > 1, then the model is worse than the naive forecast because on average, the forecast error is greater than the naive forecast error
    

    #subtract predictinos from actuals to get residuals

    #format results
    results = format_results(country, model_name, target, predictions, y_actual)

    results.meta['train_start'] = data.index[0].strftime("%Y-%m-%d")
    results.meta['train_end'] = data.index[len(Y_tr_array)-1].strftime("%Y-%m-%d")
    results.meta['lags'] = "N/A"
    results.meta['features'] = X_tr.columns.tolist() if X_tr is not None else "N/A - No Exogenous Variables"
    results.meta['model'] = model_fit
    results.meta['aic'] = round(model_fit.aic,2)
    results.meta['mape'] = round(mape,2)
    results.meta['train_rmse'] = round(np.sqrt(model_fit.mse),2)
    results.meta['params'] = params
    results.meta['model_name'] = model_name

    #do white noise test for residuals
    # BEGIN: ed8c6549bwf9
    #do white noise test for residuals
    from statsmodels.stats.diagnostic import acorr_ljungbox

    #calculate residuals
    residuals = np.array(predictions) - np.array(y_actual)

    

    #create new attribute for results called train
    if X_tr is not None:
        results.train = {}
        results.train['X_tr'] = X_tr
        results.train['X_ts'] = X_ts

    return results


def var_model (data, country, target, params, predictors = None, walk_forward = False, log_transform = False, pt = None, walkforward = False):
    '''
    Build VAR model.

    Parameters
    ----------
    data : pandas dataframe
        Dataframe containing the data to be used for the forecast model.
    country : str
        Country of interest.    
    target : str
        Target variable of interest.
    predictors : list
        List of predictor variables to be used in the model other than target variable.

    Returns
    -------
    results : dict
        Dictionary containing the results of the forecast model.

    '''
    model_name = "VAR"
    X_tr, Y_tr, X_ts, Y_ts = prepare_data(data, target, predictors)

    # concanate X_tr to Y_tr
    var_Y = pd.concat([Y_tr, X_tr], axis=1)

    # MAKE Y_tr have shape 1Xn
    Y_tr_array = Y_tr.values.reshape(len(Y_tr),1)

    c = "VAR_"+ country

    # check params for country
    with open(r'C:\Users\tfarotimi\Documents\CEPAL\Inflation Forecasting\dat/forecast_params.yaml', 'r') as file:
        parameters = yaml.load(file, Loader=yaml.FullLoader)

    # ADF Test on 'YoY Increase Inflation'
    all_p_values = []
    diff = 0
    differenced = var_Y

    for col in var_Y.columns:
        p_value = adfuller_test(var_Y[col])
        all_p_values.append(p_value)

    # if any p-value is greater than 0.05, difference the series
    if np.any(np.array(all_p_values) > 0.05):
        differenced = var_Y.diff().dropna()
        print("Difference the series")
        diff = 1

        # check if stationary after differencing
        all_p_values = []
        for col in differenced.columns:
            p_value = adfuller_test(differenced[col])
            all_p_values.append(p_value)

        # if any p-value is greater than 0.05, difference the series
        if np.any(np.array(all_p_values) > 0.05):
            differenced = differenced.diff().dropna()
            print("Difference the series")
            diff = 2

    # fit model
    if diff == 0:
        model = VAR(var_Y)
    else:
        model = VAR(differenced)

    try:
        x = model.select_order(maxlags=12)
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

    p = x.selected_orders['aic']

    # add params to yaml file for the country if it exists, else create new entry
    parameters[c] = {}
    parameters[c]['order'] = int(p)

    # write to yaml file
    with open(r'C:\Users\tfarotimi\Documents\CEPAL\Inflation Forecasting\dat/forecast_params.yaml', 'w') as file:
        yaml.dump(parameters, file)

    if (walkforward):
        target = "YoY Increase Inflation"
        model_name = model_name + " - Walk Forward"

        start = int(0.5 * len(data))
        X_tr = data.drop(columns=target).iloc[:start]
        X_ts = data.drop(columns=target).iloc[start:]

        Y_tr = data[target].iloc[:start]
        Y_ts = data[target].iloc[start:]

        # concanate X_tr to Y_tr
        var_Y = pd.concat([Y_tr, X_tr], axis=1)
        differenced = var_Y

    


        predictions = pd.DataFrame()
        tqdm._instances.clear()
        rmse_list = []
        # run model on testing data
        for i in tqdm(range(start-11), desc="Testing "+model_name, position=0, leave=True):
            next_X = X_ts.iloc[i:i+12]
            next_y = Y_ts.iloc[i:i+12]
            var_Y = pd.concat([Y_tr, X_tr], axis=1)
            # fit model on history and make a prediction
            model_fit, pred = step_forecast("var", X_tr, Y_tr, next_X,counter = i, n_steps = start - 11, params = {"order": int(p), "diff": diff})
            # store forecast in list of predictions
            #add ored 
            # add actual observations to history for the next loop
            X_tr = X_tr.append(X_ts.iloc[[i]])
            Y_tr = Y_tr.append(Y_ts.iloc[[i]])

            if diff == 1: 
               differenced = var_Y.diff().dropna()
               wf_preds = invert_transformation(var_Y, pred, diff = diff)

            elif diff == 2:
               differenced = var_Y.diff().dropna()
               differenced = differenced.diff().dropna()
               wf_preds = invert_transformation(var_Y, pred, diff = diff)
            
            else:
                wf_preds = pd.DataFrame(pred)

                
            #calculate rmse
            rmse = np.mean((wf_preds.iloc[:,0] - next_y)**2)**.5
            rmse_list.append(rmse)

        #get mean rmse
        mean_rmse = np.mean(rmse_list)
        #get std rmse
        std_rmse = np.std(rmse_list)

        print("Mean RMSE: ", mean_rmse)


        #plot rmse_list
        plt.plot(rmse_list)
        plt.title("RMSE for VAR Walk Forward")
        plt.xlabel("Months")
        plt.ylabel("RMSE")

        #show smoothed plot
        plt.plot(pd.Series(rmse_list).rolling(window=3).mean())




        plt.show()

        predictions = pd.DataFrame(wf_preds)
        y_actual = pd.DataFrame(next_y)

    else:
        # fit model
        if diff == 0:
            model = VAR(var_Y)
            model_fit = model.fit(p)
            preds = pd.DataFrame(model_fit.forecast(var_Y.values, steps=16), index = Y_ts.index,columns=var_Y.columns)

        elif diff == 1:
            model_fit = model.fit(p)
            preds = pd.DataFrame(model_fit.forecast(differenced.values, steps=16), index = Y_ts.index,columns=var_Y.columns + '_1d')

        elif diff == 2:
            model_fit = model.fit(p)
            preds = pd.DataFrame(model_fit.forecast(differenced.values, steps=16), index = Y_ts.index,columns=var_Y.columns + '_2d')
        elif diff == 3:
            model_fit = model.fit(p)
            preds = pd.DataFrame(model_fit.forecast(differenced.values, steps=16), index = Y_ts.index,columns=var_Y.columns + '_3d')
        elif diff == 4:
            model_fit = model.fit(p)
            preds = pd.DataFrame(model_fit.forecast(differenced.values, steps=16), index = Y_ts.index,columns=var_Y.columns + '_4d')

        if diff == 0:
            predictions = pd.DataFrame(preds)
        else: 
            predictions = invert_transformation(var_Y, preds, diff = diff)

        #plot impulse response for 'YoY Increase Inflation'
        irf = model_fit.irf(36)

        #plot fecv

        fecv = model_fit.fevd(36)

        

  






    # get first column of predictions and actuals
    predictions = predictions.iloc[:,0]
    #make predictions a dataframe
    #calculate mape

    #if log_transform
    if log_transform:
        #transform data
        y_actual = pd.DataFrame(Y_ts)

        pred_arr = pt.inverse_transform(np.array(predictions).reshape(-1,1)).squeeze()
        predictions = pd.DataFrame(pred_arr)

        actual_arr = pt.inverse_transform(np.array(Y_ts).reshape(-1,1)).squeeze()
        y_actual = pd.DataFrame(actual_arr, index = Y_ts.index) 
   
        
        # pred_arr = (20 * np.exp(predictions) + 1) / (1 + np.exp(predictions))
        # predictions = pd.DataFrame(pred_arr)

        # actual_arr = (20 * np.exp(y_actual) + 1) / (1 + np.exp(y_actual))
        # y_actual = pd.DataFrame(actual_arr, index = Y_ts.index)
    else:
        y_actual = pd.DataFrame(Y_ts)




    # reshape y_actual to be 1D
    if (walkforward == False):
        variability = "N/A"

        y_actual = y_actual.iloc[:,0]

        residuals = np.array(predictions) - np.array(y_actual)

        mape = 100 * np.mean(np.abs(residuals/np.array(y_actual)))

        #make series
        y_actual = pd.Series(y_actual)
    else:
        mape = 100 * np.mean(np.abs((y_actual.iloc[:,0] - predictions)/y_actual.iloc[:,0]))#divide each forecast error by the actual value and take the average of the absolute values


    # calculate mase with out-of-sample naive forecast
     #calculate residuals
   
    #calculate mape
    


    # if mase > 1, then the model is worse than the naive forecast because on average, the forecast error is greater than the naive forecast error

    # subtract predictions from actuals to get residuals

    # format results
    results = format_results(country, model_name, target, predictions, y_actual)

    results.meta['train_start'] = data.index[0].strftime("%Y-%m-%d")
    results.meta['train_end'] = data.index[len(Y_tr_array)-1].strftime("%Y-%m-%d")
    results.meta['lags'] = "N/A"
    results.meta['features'] = X_tr.columns.tolist() if X_tr is not None else "N/A - No Exogenous Variables"
    results.meta['model'] = model_fit
    results.meta['aic'] = round(model_fit.aic,2)
    results.meta['mape'] = round(mape,2)
    results.meta['train_rmse'] = round(np.mean((predictions - y_actual)**2)**.5,2)  # RMSE
    results.meta['diff'] = diff
    results.meta['params'] = p
    results.meta['differenced'] = differenced if diff > 0 else "N/A"
    results.meta['variability'] = variability or "N/A"
    results.meta['irf'] = irf
    results.meta['fecv'] = fecv
    results.meta['model_name'] = model_name


    #do white noise test for residuals
    # BEGIN: ed8c6549bwf9
    #do white noise test for residuals
    from statsmodels.stats.diagnostic import acorr_ljungbox

   

    #create new attribute for results called train
    if X_tr is not None:
        results.train = {}
        results.train['X_tr'] = X_tr
        results.train['X_ts'] = X_ts

    return results    

def cnn_model (data, country, target, predictors = None, walk_forward = False, log_transform = False, pt = None, walkforward = False):
    '''
    Build CNN model.

    Parameters
    ----------
    data : pandas dataframe
        Dataframe containing the data to be used for the forecast model.
    country : str
        Country of interest.    
    target : str
        Target variable of interest.
    predictors : list
        List of predictor variables to be used in the model other than target variable.

    Returns
    -------
    results : dict
        Dictionary containing the results of the forecast model.

    '''
    
    model_name = "CNN"

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


    import numpy as np
    from keras.layers import MaxPooling1D, Conv1D, Dense, Flatten

    # Prepare supervised learning samples from the full normalized data before splitting
    all_values = norm_data['YoY Increase Inflation'].values
    X_all, y_all = [], []
    for i in range(n_lags, len(all_values)):
        X_all.append(all_values[i - n_lags:i])
        y_all.append(all_values[i])

    X_all, y_all = np.array(X_all), np.array(y_all)


    # Split into train and test sets using train_size
    X_train, y_train = X_all[:train_size - n_lags], y_all[:train_size - n_lags]
    X_test, y_test = X_all[train_size - n_lags:], y_all[train_size - n_lags:]


    # Reshape for CNN input
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    y_train = y_train.reshape((y_train.shape[0], 1))
    y_test = y_test.reshape((y_test.shape[0], 1))

    n_val = 12
    X_val, y_val = X_train[-n_val:], y_train[-n_val:]

    #reshape X_train
    X_train_adj, y_train_adj = X_train[:-n_val], y_train[:-n_val]




    # Build and train CNN model
    # model = Sequential()
    # model.add(Conv1D(64, 3, activation='relu', input_shape=(n_lags, 1)))
    # model.add(Dropout(0.2))
    # model.add(Conv1D(32, 3, activation='relu'))
    # model.add(Flatten())
    # model.add(Dense(64, activation='relu'))
    # model.add(Dense(12))
    #install tcn

    import tensorflow as tf
    from tcn import TCN
    from tensorflow.keras.optimizers import Adam
    #import input layer
    from tensorflow.keras.layers import InputLayer, Dense
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Input, Dense, LSTM, Bidirectional, Dropout, LayerNormalization, MultiHeadAttention, GlobalAveragePooling1D, Add
    from tensorflow.keras.optimizers import Adam
  

    model = Sequential()
    input_shape = (X_train.shape[1], 1)  # Define input shape for the InputLayer
    output_steps = 12           # Define output_steps, or set as needed
    model.add(InputLayer(input_shape=input_shape))
    model.add(TCN(nb_filters=64, kernel_size=3, dilations=[1, 2, 4, 8], activation='relu', dropout_rate=0.2))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(output_steps))  # output_steps = number of forecast steps (e.g., 12)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

    # inputs = Input(shape=input_shape)
    # x = Bidirectional(LSTM(64, return_sequences=True))(inputs)
    # x = Dropout(0.2)(x)
    # x = Bidirectional(LSTM(32))(x)
    # x = Dense(50, activation='relu')(x)
    # outputs = Dense(output_steps)(x)
    # model = Model(inputs, outputs)
    # model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

    # num_heads = 4  # Number of attention heads
    # key_dim = 16  # Dimension of the attention keys
    # inputs = Input(shape=input_shape)
    # x = LayerNormalization()(inputs)
    # attention_output = MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)(x, x)
    # x = Add()([x, attention_output])
    # x = LayerNormalization()(x)
    # x = GlobalAveragePooling1D()(x)
    # x = Dense(64, activation='relu')(x)
    # x = Dropout(0.1)(x)
    # outputs = Dense(output_steps)(x)
    # model = Model(inputs, outputs)
    # model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')



    def smape(y_true, y_pred):
        denominator = (tf.abs(y_true) + tf.abs(y_pred)) / 2.0
        diff = tf.abs(y_true - y_pred) / denominator
        return tf.reduce_mean(diff)

    #model.compile(optimizer=Adam(), loss=smape)

    model.fit(X_train, y_train, epochs=200, verbose=0, batch_size=10, validation_data = (X_val, y_val))

    cnn_pred = []
    history = [x for x in X_train]
    for i in range(y_test.shape[0]):
        h_data = np.array(history)
        # Reshape data to be 3D [samples, timesteps, features]
        h_data = h_data.reshape((h_data.shape[0]*h_data.shape[1], 1))  # Reshape to [1, n_lags, 1]
        # Predict the next value using the last n_lags values from training set
        input_x = h_data[-n_lags:,0]
        input_x = input_x.reshape((1, n_lags, 1))
          # Last n_lags values from training set
        next_pred = model.predict(input_x, verbose=0)[0][0]
        cnn_pred.append(next_pred)
        history.append(X_test[i])

    
    cnn_pred = np.array(cnn_pred)
    # Reshape predictions to match the original data shape
    cnn_pred = cnn_pred.reshape(-1, 1)
    # Rescale predictions back to original scale
    cnn_pred = scaler.inverse_transform(cnn_pred)
    # Rescale y_test back to original scale
    y_test = scaler.inverse_transform(y_test)
    # Reshape predictions and actual values to dataframe with .index
    cnn_pred = pd.DataFrame(cnn_pred, index=data.index[train_size:], columns=['YoY Increase Inflation'])
    y_test = pd.DataFrame(y_test, index=data.index[train_size:], columns=['YoY Increase Inflation'])
    #format results
    results = format_results(country, model_name, target, cnn_pred, y_test)

    #add meta information to results
    results.meta['train_start'] = data.index[0].strftime("%Y-%m-%d")
    results.meta['train_end'] = data.index[len(y_train)-1].strftime("%Y-%m-%d")
    results.meta['lags'] = n_lags
    results.meta['features'] = X_train.shape[1]
    results.meta['model'] = model
    results.meta['aic'] = round(model.evaluate(X_train, y_train),2)
    results.meta['mape'] = round(np.mean(np.abs((cnn_pred - y_test)/y_test) * 100),2) 
    results.meta['train_rmse'] = round(np.mean((cnn_pred - y_test)**2)**.5,2)  # RMSE
    results.meta['params'] = n_lags	
    results.meta['train_size'] = train_size
    results.meta['test_size'] = len(y_test)
    results.meta['scaler'] = scaler
    results.meta['X_train'] = X_train
    results.meta['X_test'] = X_test
    results.meta['y_train'] = y_train
    results.meta['y_test'] = y_test
    results.meta['X_all'] = X_all
    results.meta['y_all'] = y_all
    results.meta['model_name'] = model_name
    
    return results

def recursive_forecast_rf_strict(model, full_target_series, future_exog=None, lags=12, horizon=12):

    """
    Perform a fully recursive 12-step forecast using a trained Random Forest model,
    dynamically rebuilding lag features after each prediction.

    Parameters:
    - model: trained RandomForestRegressor (already fit to training data)
    - full_series: pd.Series of inflation (YoY) with datetime index (used to generate lag features)
    - lags: number of lags to use in feature generation
    - horizon: forecast horizon (e.g., 12 months)

    Returns:
    - forecast: pd.Series of 12-step-ahead forecasted values
    """

    series = full_target_series.copy()
    series.index.name = 'Period'

    future_exog.index = pd.to_datetime(future_exog.index).to_period('M').to_timestamp('M')
    future_exog.index.name = 'Period'


      # Ensure index is datetime
    print("future exog shape:", future_exog.shape)


    future_exog.index = pd.to_datetime(future_exog.index)  # Ensure future_exog index is datetime
    future_exog.index = future_exog.index.to_period('M').to_timestamp(how='start')  # Ensure future_exog index is monthly

    preds = []

    for step in range(horizon):
    # Step 1: Rebuild lagged features from current series
        lagged_df = time_delay_embedding(series, n_lags=lags, horizon=0)
        latest_X = lagged_df.iloc[[-1]]  # most recent lags only

      

        #write future_exog to pickle 
        # Only pickle if future_exog is not None and not empty
        if future_exog is not None and not future_exog.empty:

            #pickle full_data[target]
            series.to_pickle(f"series.pkl")
            print("target Pickle object created")
         
            future_exog.to_pickle(f"future_exog_step_{step}.pkl")
            print("Pickle object created")
        else:
            print("future_exog is None or empty, skipping pickle.")

            
        # Step 2: Add corresponding future exog values
        if future_exog is not None:
            # pdb.set_trace()
            forecast_month = series.index[-1] + pd.DateOffset(months=1)

            print("forecast_month:", forecast_month)
            print("future_exog index dtype:", future_exog.index.dtype)
            print("forecast_month in future_exog.index:", forecast_month in future_exog.index)

            print("future_exog index dtype:", future_exog.index.dtype)
            print("forecast_month type:", type(forecast_month))
            print("forecast_month in future_exog.index:", forecast_month in future_exog.index)

            

            if forecast_month in future_exog.index:
                exog_row = future_exog.loc[[forecast_month]]  # keep it as a row
                exog_row.index = latest_X.index
                latest_X = latest_X.join(exog_row, how='left')
            else:
                print(f"⚠️ No exog data found for {forecast_month}. Filling with NaN.")
                latest_X = latest_X.reindex(columns=list(latest_X.columns) + list(future_exog.columns))

       #print("latest_X:", latest_X)


        # Step 2: Add trend, month dummies, etc.
        latest_X["trend"] = len(series)
        latest_X["month"] = series.index[-1].month
        latest_X = pd.get_dummies(latest_X, columns=["month"], drop_first=True)

        # Ensure dummy columns align with training set (e.g., month_2..month_12)
        for m in range(2, 13):
            col = f"month_{m}"
            if col not in latest_X.columns:
                latest_X[col] = 0  # add missing month dummies


        # Step 3: Add rolling stats (be conservative on window size to avoid NaN)
        for w in [3, 6]:
            latest_X[f"roll{w}_mean"] = series[-w:].mean()
            latest_X[f"roll{w}_std"] = series[-w:].std()

        latest_X["mom"] = series.pct_change().iloc[-1]
        latest_X["yoy"] = series.pct_change(12).iloc[-1] if len(series) > 12 else 0

        #drop target 
        latest_X = latest_X.drop(columns=[full_target_series.name], errors='ignore')

        #sort latest_X columns to match training set
        latest_X = latest_X.reindex(sorted(latest_X.columns), axis=1)

        # Reorder columns if necessary
        #latest_X = latest_X[sorted(latest_X.columns)]

        # Step 4: Predict and append
        y_hat = model.predict(latest_X)[0]
        next_date = series.index[-1] + pd.DateOffset(months=1)
        series.loc[next_date] = y_hat
        preds.append((next_date, y_hat))

    # Return as Series
    forecast = pd.Series({date: val for date, val in preds})
    return forecast





