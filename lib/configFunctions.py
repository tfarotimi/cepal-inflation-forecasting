# Author(s): Albert Bredt
# Last update- 11/05/2023 by Inflation Forecasting Farotimi

# This code contains a set of functions to create or edit config files:
# - createConfigFromExcel

# Importing necessary modules
import pandas as pd
import yaml

########################################################################################################################
# createConfigFromExcel - This function creates a yml config file from an excel file. The highlighted loop is adjusted to
# the example file "exampleConfig.xls" and needs to be adjusted to the specific input and the desired structure of the output.
# Latest update: 23.04.2021 by Albert Bredt

# Parameters:
    # CNAME (string): Name of the saved config file
    # SOURCE (string, optional): address of the source file (.xls)
    # SAVEPATH (string, optional): path where the config should be stored

# Example:
    # createConfigFromExcel('Test', SOURCE='/Dev/exampleConfig.xls') will open the exampleConfig file, translate it to a dictionary and save it as 'configTest.yml'
    # the specified CNAME in the SAVEPATH

# Defining the function
def createConfigFromExcel(inst, SOURCE='', SAVEPATH=''):
    dict = {}
    # Open excel file
    dfs = pd.read_excel(SOURCE, sheet_name=None)
    # This part reads out the excel file and creates the dictionary structure
    ############### Needs to be adapted to the specific excel file and the desired output ##############################
    if inst == 'CHL_CB':
        for i, j, k in zip( dfs[inst]['Variable_name'],dfs[inst]['Code'], dfs[inst]['Series_name']):
            dict[i] = {"seriesID": j, "seriesName": k,"seriesFormat":"DD-MM-YYYY"}
    elif inst == 'MEX_CB':
        for i, j, k in zip( dfs[inst]['Variable_name'],dfs[inst]['Code'], dfs[inst]['Series_name']):
            dict[i] = {"seriesID": j, "seriesName": k,"seriesFormat":"DD/MM/YYYY"}
    elif inst == 'ARG_ALL':
        for i, j, k in zip( dfs[inst]['Variable_name'],dfs[inst]['Code'], dfs[inst]['Series_name']):
            dict[i] = {"seriesFormat":"YYYY-MM-DD","seriesFrequency": 'M',"seriesID": j}
    elif inst == 'BOL_NSO':
        for i, j, k,l in zip( dfs[inst]['Variable_name'],dfs[inst]['Code'], dfs[inst]['Group'], dfs[inst]['Series']):
            dict[i] = {"seriesID":
                       {"group":k, "series":l},
                       "seriesFormat":"MMM-YY_es",
                       "seriesFrequency": 'M'}
    elif inst == 'BRA_CB':
        for i, j, k in zip(dfs[inst]['Variable_name'],dfs[inst]['Code'], dfs[inst]['Series_name']):
            dict[i] = {"seriesID": str(j),"seriesFrequency": 'M', "seriesFormat":"DD/MM/YYYY"}
    elif inst == 'PER_CB':
        for i, j, k in zip( dfs[inst]['Variable_name'],dfs[inst]['Code'], dfs[inst]['Series_name']):
            dict[i] = {"seriesID": str(j),"seriesFrequency": 'M', "seriesFormat":"MMM.YYYY_es"}




            
    # Saves the dictionary as .yml file
    with open(str(SAVEPATH) + '/configAPI_' + inst + '.yml', 'w') as file:
        yaml.dump(dict, file)
