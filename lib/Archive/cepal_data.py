import pandas as pd 
import numpy as np


def get_CEPAL_Data(filename):
    
    FILEPATH = R"C:\Users\tfarotimi\Documents\CEPAL\Inflation Forecasting\dat\CEPALSTAT_Raw\\"

    cepalstat = pd.read_excel(FILEPATH + filename, sheet_name=['data']).get('data')
    cepal_data = {}
    
    for country in cepalstat['Country__ESTANDAR'].unique():
        #create a dataframe for each country
        cepal_data[country] = cepalstat[cepalstat['Country__ESTANDAR'] == country]

        

        #create a date column
        cepal_data[country]['Date'] = cepal_data[country]['Months'] + ' ' + cepal_data[country]['Years__ESTANDAR'].astype(str)
        cepal_data[country]['Date'] = pd.to_datetime(cepal_data[country]['Date'],format='%B %Y')

        #set date as index
        cepal_data[country].set_index('Date', inplace=True)
        cepal_data[country].sort_values(by='Date', inplace=True)
    
        #drop columns that are not needed
        cepal_data[country].drop(columns=['indicator', 'Country__ESTANDAR', 'unit','notes_ids','source_id','Years__ESTANDAR', 'Months'], inplace=True)

        #only keep rows with 'value' >= 100 (reset index)
        index_reset = cepal_data[country][cepal_data[country]['value'] == 100]
        if index_reset.empty:
            
            if (country[0:7] == 'Bolivia'):
                cepal_data[country] = cepal_data[country][cepal_data[country].index >= '1986-01-01']
            elif (country[0:9] == 'Venezuela'):
                cepal_data[country] = cepal_data[country][cepal_data[country].index >= '2019-01-01']
            else:
                cepal_data[country]=cepal_data[country]
        else:
            if (country == 'Peru'):
                cepal_data[country] = cepal_data[country][cepal_data[country].index >= index_reset.index[-1] - pd.DateOffset(months=24)]
            else:
                cepal_data[country] = cepal_data[country][cepal_data[country].index >= index_reset.index[-1]]

        #calculate %change year over year for 'value'column
        cepal_data[country]['value'] = cepal_data[country]['value'].pct_change(periods=12)*100
        cepal_data[country].rename(columns={'value': 'headline_cpi_yoy_change'}, inplace=True)

       

        #drop rows with null values
        cepal_data[country].dropna(inplace=True)


       

        #remove keys with no values (countries with no data)
        if cepal_data[country].empty:
            del cepal_data[country]

    

    
    return cepal_data