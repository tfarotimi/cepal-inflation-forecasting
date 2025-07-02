# BEGIN: ed8c6549bwf9
    #do white noise test for residuals
    from statsmodels.stats.diagnostic import acorr_ljungbox

    #calculate residuals
    residuals = np.array(predictions) - np.array(y_actual)

    #perform Ljung-Box test for autocorrelation in residuals
    lbvalue, pvalue = acorr_ljungbox(residuals, lags=12)

    if pvalue.min() < 0.05:
        print("Residuals are autocorrelated")
    else:
        print("Residuals are not autocorrelated - autocorrelated at:" pvalue[pvalue < 0.05])
# END: ed8c6549bwf9