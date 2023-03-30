# project2

# SMA Signal Creation

## Running the notebook:
1. Navigate to the directory containing main_app.ipynb using the console. <br>
2. Ensure voila is installed using pip install voila. <br>
3. Type voila main_app.ipynb into the console. <br>

## Imports: 
The following imports are required to run the SMA, OBV and RSI trading strategies:
import os
from dotenv import load_dotenv
import alpaca_trade_api as tradeapi
import numpy as np
import hvplot.pandas
import matplotlib.pyplot as plt
import pandas as pd
### For the talib import you must install it first --> Open a terminal tab and type "pip install TA-Lib". Then import talib in imports cell.
#import talib (had trouble with the installs and could not run the RSI)
from technical import indicators

## Create and Load env file
1. A .env file must be create with keys to access Alpaca API. Consider both API key and secret key.
2. Insert the API using getenv
3. Create access to Alpaca with the REST function
4. Insert the ticker of the commidity you are going to use. In our case we selected Soybean Oil (SOYB). Consider the timeframes to evaluate the trading strategy.
5. Create the dataframe you're going to use by using the get_bars function. Display the data to make sure it runs.
6. Clean the data to align required information to test the trading strategy.

## Develop SMA Strategy:
1. Define short and log windows.
2. Add signals for both SMAs on the data frame.
3. Create de trading signals based off of 0 / 1.
4. Define both entry and exit and include the diff() function.
5. Print dataframe to verify.

## Plot Trading Strategy
1. Create exit and entry points in a way they're visually present in the graph to identify both executions.
2. Identify closing prices for positions.
3. Visualize moving averages.
4. Create overlay plot.
5. Visualize.

# Backtest SMA Strategy Using Predefined Capital
1. Define a capital amount.
2. Define number of shares to enter a position.
3. Idenfity when a position is "longed" or "shorted".
4. Find the ROI.
5. Review portfolio liquidity.
6. Determine all assets in portfolio (cash and equity in position/s)
7. Figure out daily returns and commulative returns.
8. Review dataframe.

## Plot Portfolio Behavior in Time
1. Identify entry / exit.
2. Define total portfolio value.
3. Overlay plots.

# On-balance Volume (OBV) Signal Creation
1. Creat column for the daily price change calculations.
2. Calculations for the On-Balance Volume and store it under 'OBV'.
3. Create a function for trading signals to be generated according to the On-Balance Volume indicator. Include long and short thresholds and generate trading signals based on OBV.
4. Plot price-action along with trading signals.
5. Plot OBV along with quantiles

# Backtest OBV Strategy Using Predefined Capital
1. Define a capital amount.
2. Define number of shares to enter a position.
3. Define trading signals and threshold
4. Enter a long or short and idenfity when a position is "longed" or "shorted".
5. Find the ROI.
6. Review portfolio liquidity.
7. Determine all assets in portfolio (cash and equity in position/s)
8. Figure out daily returns and commulative returns.
9. Review dataframe.
## Conclusion: The SMA performed better than the OBV; therefore, we selected the SMA signal to move on with the project.

# RSI Signal Creation
## Import talib library in order to be able to run the code
### Open a terminal tab and type "pip install TA-Lib". Then import talib in imports cell. Run the code from above.
## NOTE: Please note we had trouble running this code due to talib installs not working. Therefore, this indicator was disqualified for the signal creation process selection in this project. Nevertheless, the code will be vissible and should run if talib library runs appropriately.

# Support vector machine model
## Imports:
from os import getenv
from dotenv import load_dotenv
from alpaca_trade_api.rest import REST, TimeFrame, TimeFrameUnit
import pandas as pd
from datetime import date, timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.svm import SVC
import numpy as np


## Support Vector Machine Classification Model
1. The model trains on data from April 2021 to March 2022 including SMA signals, previous closing prices, volume, vwap, and trade counts.
2. The model is tested on its accuracy in predicting the buy or sell signals from April 2022 to March 2023

## Support Vector Machine Regression Model
1. The model trains on data from April 2021 to March 2022 including previous closing prices, volume, vwap and trade counts.
2. The model is tested on its ability to predict closing prices from April 2022 to March 2023
3. A plot comparing the model's predicted closing prices to the actual closing prices is generated.
4. R^2 and RMSE for the model are calculated.

# XGBOOST model
## Imports:
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error

## XGBOOST Classification Model
1. The model uses the XGBClassifier from XGBOOST library.
2. It specifies the objective parameter as "multi:softmax", which is a common choice for classification problems.
3. The num_class parameter is set to 3, which indicates that there are three classes in the problem.
4. The model performed poorly according to r^2 and RMSE scores, leading us to optimize the model in the next step.

## XGBOOST Regression Model
1. The objective parametere for the model is set to 'reg:squarederror', which emphasizes that the model is trained to minimize the mean squared error.
2. The target variable y_train is continuous, which further verifies that the model is a regression model.
3. The optimal hyperparameters were utilized through trial and errors and based on documented best practices for XGBOOST algorithm.
4. The optimized model generated the best r^2 and RMSE compared to the intitial XGBOOST model. 

# Neural Networks
## Imports:
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

## Neural Network Classification Model
1. The model takes features from the SMA strategy to predict the buy/sell signal. 
2. The neural network model is trained in the "MB Classifier.ipynb" notebook with two hidden layers and one output layer.
3. The activation function "Sigmoid" was found to produce the best results with the training data. 
4. To use the models weights, import "nn_classifier.h5"

## Neural Network Regression Model
1. The model attempts to predict the future closing price of SOYB.
2. The neural network model is trained in the "MB Regression.ipynb" notebook with one hidden layers and one output layer.
3. The activation function "Linear" was found to produce the best results with the training data.
4. To use the models weights, import "nn_regression.h5"
