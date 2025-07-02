#add lib to path
import sys
sys.path.append(r'C:\Users\tfarotimi\Documents\CEPAL\Inflation Forecasting\src')
sys.path.append(r"C:\Users\tfarotimi\Documents\CEPAL\Inflation Forecasting\lib")

#import libraries 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import plotly.offline as pyo
from plotly.subplots import make_subplots
pyo.init_notebook_mode()
import nbformat

#tqdm
from tqdm import tqdm

#scikit-learn
from sklearn.metrics import mean_squared_error,r2_score



#import functions from modules
from models import mean_forecast, naive_forecast, univar_ARDL, multivar_ARDL, random_forest_w_lags, sarimax, arima
from helper_functions import load_data, prepare_data, format_results,step_forecast, time_delay_embedding
from dashboard import model_compare, r2_compare, error_compare
from residuals_analysis import analyze_residuals
from API_data_request import get_API_data
from forecast import get_forecast 



#import modules
import importlib
import models
import helper_functions
import dashboard
import residuals_analysis
import API_data_request
import forecast

#reload modules to ensure that changes made to them are reflected in the notebook
importlib.reload(models)
importlib.reload(helper_functions)
importlib.reload(dashboard)
importlib.reload(residuals_analysis)
importlib.reload(API_data_request)
importlib.reload(forecast)

#set path to data folder
import warnings
warnings.filterwarnings('ignore')

