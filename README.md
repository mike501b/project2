# project2

# SMA Signal Creation

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

