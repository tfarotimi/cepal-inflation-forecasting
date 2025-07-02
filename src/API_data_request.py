import os
import sys
from pathlib import Path
from yaml import safe_load
from datetime import datetime
import pandas as pd

 #add path for Inflation Forecasting/lib folder to enable imports
CODEPATH = str(Path(__file__).parents[1] / 'lib')
sys.path.append(os.path.abspath(CODEPATH))

#ADD path for Code folder to enable imports
DOWNLOAD_API_CODEPATH= str(Path(__file__).parents[3]/'Code')
sys.path.append(os.path.abspath(DOWNLOAD_API_CODEPATH))


#import necessary functions 
from configFunctions import createConfigFromExcel
from mainDownloadTS_API import mainDownloadTS_APIs
from downloadTS_APIs import downloadTS_APIs, APIs

#set CONFIGPATH equal to the path of the config file
CONFIGPATH = str(Path(__file__).parents[3] / 'Configs')

today = datetime.today().strftime('%Y-%m-%d')
month = datetime.today().strftime('%B')
year = datetime.today().strftime('%Y')


def get_API_data(institutions, variables):

    institutions = institutions if institutions else list(APIs.keys())

    ROOT_FOLDER = Path(__file__).parents[0].parents[0].parents[0].parents[0]

    #set path to excel file with series list
    CONFIG_SOURCE = ROOT_FOLDER / 'Dev' / 'Inflation Forecasting' / 'dat' /'config_sources'/ 'API_Config_Source.xlsx'

    #set path to save config file
    CONFIG_SAVEPATH = ROOT_FOLDER /'Dev'/'Inflation Forecasting'/ 'lib' / 'API_Configs/'

    #create folder for year and month if it does not exist
    if not os.path.exists(str(ROOT_FOLDER) + "/" + "Dev/Inflation Forecasting/dat/" + year + "/" + month):
        os.makedirs(str(ROOT_FOLDER) + "/" + "Dev/Inflation Forecasting/dat/" + year + "/" + month)



    #create config file
    for inst in institutions:
        createConfigFromExcel(inst, SOURCE=CONFIG_SOURCE, SAVEPATH=CONFIG_SAVEPATH)

        api_data = mainDownloadTS_APIs(varlist=variables, institutions=[inst], region = "", sDate = "", eDate = "" ,  frequency= "", method= "", name = inst+"_Inflation_Time_Series_Data", ONEFILE = 1, FILEPATH = str(ROOT_FOLDER) + "/" + "Dev/Inflation Forecasting/dat/" + year + "/" + month + "/", CONFIGPATH = str(CONFIG_SAVEPATH)+"/")    


    #get API data using config file
