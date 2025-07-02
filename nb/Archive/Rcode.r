f <- function (X) {

            #load libraries
            library(tseries)
            library(forecast)
            library(Metrics)
            library(rugarch)
      
            #create test and train sets for ARIMA-GARCH model - test is last 12 months of X
            X <- as.ts(X)
            train <- head(X, -12)
            test <- tail(X, 12)
    
            #do arima grid search to get optimal arima parameters 
            print("arima grid search")
            arima_model = auto.arima(X, D=1, ic = "aic", trace = TRUE, stepwise = TRUE, seasonal=TRUE, approximation = FALSE, allowdrift = TRUE, allowmean = TRUE, xreg = NULL, test = "kpss")

            #analyze residuals of arima model
            print("residuals analysis")
            resids <- residuals(arima_model)
            
            print("white noise test")
            #check if residuals are white noise
            white_noise_test = tryCatch(
                  Box.test(resids, lag = 20, type = "Ljung-Box"), error=function(e) e, warning=function(w) w
            )$p.value
            print(white_noise_test)
            
            if (white_noise_test < 0.05) {
                  print("arima residuals are not white noise")
            }
            else {
                  print("arima residuals are white noise")
            }

            #check heteroskedasticity of residuals
            het_test <- tryCatch(
                  het.test(resids), error=function(e) e, warning=function(w) w
            )$p.value
            
            if (het_test < 0.05) {
                  print("arima residuals are heteroskedastic")
            }
            else {
                  print("arima residuals are homoskedastic")
            }

            #check normality of residuals
            norm_test <- tryCatch(
                  shapiro.test(resids), error=function(e) e, warning=function(w) w
            )$p.value

            if (norm_test < 0.05) {
                  print("arima residuals are not normally distributed")
            }
            else {
                  print("arima residuals are normally distributed")
            }

            #check mean of residuals
            mean_test <- tryCatch(
                  t.test(resids), error=function(e) e, warning=function(w) w
            )$p.value

            if (mean_test < 0.05) {
                  print("arima residuals do not have a mean of zero")
            }
            else {
                  print("arima residuals have a mean of zero")
            }

            #check if kpss test is stationary, if so print stationary, else print not stationary
            if (kpss.test(X[,1])$p.value < 0.05) {
                  garch_X <- diff(X)
                  print("Differencing data for garch model fitting")
                  }
            else {
                  garch_X <- X
                  print("Data is already stationary")
                  }

            #get best p and q values for garch model using garch grid search
            print("garch grid search")
            garch_model = garchFit(~garch_X, data=garch_X, trace=F, cond.dist="std", include.mean=T, include.delta=T, leverage=T, include.skew=T, include.shape=T, include.omega=T, include.alpha=T, include.beta=T, include.gamma=T, include.vega=T, include.ar=T, include.ma=T, include.var=T, include.mean=T, include.skew=T, include.shape=T, include.delta=T, include.
            #get best p and q values from garch_model
            p = garch_model$order[1]
            q = garch_model$order[2]
            print("Fitting different Garch with p, q = ")
            print(p, q)




            #fit garch model to garch_X
            print ("fit garch")
            garch_X.garch <- garch(garch_X, trace=F)
            print(summary(garch_X.garch))
            #extract p and q values from garch_X.garch
            p = garch_X.garch$order[1]
            q = garch_X.garch$order[2]
            print("Fitting Garch with p, q = ")
            print(p, q)
      

              #create spec for ugarchfit with mean.model as arima_model and variance.model as garch(p,q)
            print("analyze garch residuals")

            garch_X.res <- garch_X.garch$res[-1]
            
            acf(garch_X.res, main="ACF of Residuals")
            acf(garch_X.res^2, main="ACF of Squared Residuals")

            spec = ugarchspec(
                  variance.model=list(model="sGARCH", garchOrder=c(p,q)),
                  mean.model=list(arima_model, include.mean=T),
                  distribution.model="std"
                  )
            
            #fit arima-garch model to train set
            arima_garch_train_fit <- ugarchfit(
            spec, train, solver = 'hybrid'
            )
            print(arima_garch_train_fit)
            
            #predict values for test set
            print("test set forecast arima garch:")
            test_fc <- ugarchforecast(arima_garch_train_fit, n.ahead=12,bootstrap=TRUE)
            print(test_fc)
            #get predicted values from test forecast
            test_preds <- fitted(test_fc)
            print(test_preds)
            print(summary(test_fc))
            print(test)

            #display rmse of test forecast
            print("test rmse arima")

            #calculate rmse between test_fc and test
            rmse = rmse(test_preds, test)
            print(rmse)
            #get R^2 of test forecast
            print("test r2 arima")
            r2_score = 1 - (rmse^2 / var(as.numeric(test)))
            print(r2_score)

            #forecast for next 12 months
            fc_fit = ugarchfit(
            spec, X, solver = 'hybrid'
            ) 
            
            fc_resids <- residuals(fc_fit)

            #set column name for residuals
            colnames(resids) <- 'Residuals'

            fore = ugarchforecast(fc_fit, out.sample = 12, n.ahead=12)
            plot(fore, which = 1)
            info <- infocriteria(fit)
            ind = fore@forecast$seriesFor
      
            white_noise_test = tryCatch(
                  Box.test(fc_resids, lag = 20, type = "Ljung-Box"), error=function(e) e, warning=function(w) w
            )$p.value
            print(white_noise_test)
            print("arima garch residuals analysis")

            if (white_noise_test < 0.05) {
            print("arima garch residuals are not white noise")
            }
            else {
            print("arima garch residuals are white noise")
            }

                        
            #check heteroskedasticity of residuals
            het_test <- tryCatch(
                  het.test(fc_resids), error=function(e) e, warning=function(w) w
            )$p.value

            if (het_test < 0.05) {
                  print("arima garch residuals are heteroskedastic")
            }
            else {
                  print("arima garch residuals are homoskedastic")
            }

            #check normality of residuals
            norm_test <- tryCatch(
                  shapiro.test(fc_resids), error=function(e) e, warning=function(w) w
            )$p.value

            if (norm_test < 0.05) {
                  print("arima garch residuals are not normally distributed")
            }
            else {
                  print("arima garch residuals are normally distributed")
            }

            
            tests = list(white_noise_test, het_test, norm_test)
            show(fc_fit)

            results <- list(ind, info, tests)
            return(results)
            
            #get out - rmse, r2, test_preds, forecast, volatility, test p-value
            # return(list(rmse, r2_score, test_preds, volatility, test_p_values))
            # clean up
            # in python, calculate confidence intervals, and send to excel report 

        #create spec for ugarchfit with mean.model as arima_model and variance.model as garch(p,q)
            print("analyze garch residuals")

            garch_X.res <- garch_X.garch$res[-1]
            
            acf(garch_X.res, main="ACF of Residuals")
            acf(garch_X.res^2, main="ACF of Squared Residuals")

            spec = ugarchspec(
                  variance.model=list(model="sGARCH", garchOrder=c(p,q)),
                  mean.model=list(arima_model, include.mean=T),
                  distribution.model="std"
                  )
            
            #fit arima-garch model to train set
            arima_garch_train_fit <- ugarchfit(
            spec, train, solver = 'hybrid'
            )
            print(arima_garch_train_fit)
            
            #predict values for test set
            print("test set forecast arima garch:")
            test_fc <- ugarchforecast(arima_garch_train_fit, n.ahead=12,bootstrap=TRUE)
            print(test_fc)
            #get predicted values from test forecast
            test_preds <- fitted(test_fc)
            print(test_preds)
            print(summary(test_fc))
            print(test)

            #display rmse of test forecast
            print("test rmse arima")

            #calculate rmse between test_fc and test
            rmse = rmse(test_preds, test)
            print(rmse)
            #get R^2 of test forecast
            print("test r2 arima")
            r2_score = 1 - (rmse^2 / var(as.numeric(test)))
            print(r2_score)

            #forecast for next 12 months
            fc_fit = ugarchfit(
            spec, X, solver = 'hybrid'
            ) 
            
            fc_resids <- residuals(fc_fit)

            #set column name for residuals
            colnames(resids) <- 'Residuals'

            fore = ugarchforecast(fc_fit, out.sample = 12, n.ahead=12)
            plot(fore, which = 1)
            info <- infocriteria(fit)
            ind = fore@forecast$seriesFor
      
            white_noise_test = tryCatch(
                  Box.test(fc_resids, lag = 20, type = "Ljung-Box"), error=function(e) e, warning=function(w) w
            )$p.value
            print(white_noise_test)
            print("arima garch residuals analysis")

            if (white_noise_test < 0.05) {
            print("arima garch residuals are not white noise")
            }
            else {
            print("arima garch residuals are white noise")
            }

                        
            #check heteroskedasticity of residuals
            het_test <- tryCatch(
                  het.test(fc_resids), error=function(e) e, warning=function(w) w
            )$p.value

            if (het_test < 0.05) {
                  print("arima garch residuals are heteroskedastic")
            }
            else {
                  print("arima garch residuals are homoskedastic")
            }

            #check normality of residuals
            norm_test <- tryCatch(
                  shapiro.test(fc_resids), error=function(e) e, warning=function(w) w
            )$p.value

            if (norm_test < 0.05) {
                  print("arima garch residuals are not normally distributed")
            }
            else {
                  print("arima garch residuals are normally distributed")
            }

                  
            tests = list(white_noise_test, het_test, norm_test)
            show(fc_fit)

            results <- list(ind, info, tests)
            return(results)
            
            #get out - rmse, r2, test_preds, forecast, volatility, test p-value
            # return(list(rmse, r2_score, test_preds, volatility, test_p_values))
            # clean up
            # in python, calculate confidence intervals, and send to excel report 
}

     