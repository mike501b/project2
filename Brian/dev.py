from os import getenv
from dotenv import load_dotenv
from alpaca_trade_api.rest import REST, TimeFrame, TimeFrameUnit
import pandas as pd
from datetime import date, timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.svm import SVC
import numpy as np

def get_bars_alpaca(start_date=str(date.today()-timedelta(days=365*5)),end_date=str(date.today()-timedelta(days=1)),ticker='SOYB'):
    load_dotenv()
    alpaca_api_key=getenv('ALPACA_API_KEY')
    secret_key=getenv('ALPACA_SECRET_KEY')
    if type(alpaca_api_key) == 0:
        raise Exception("No ALPACA API KEY was loaded. Please save a .env file containing an ALPACA_API_KEY in the local directory.")
    if type(secret_key) == 0:
        raise Exception("No ALPACA SECRET KEY was loaded. Please save a .env file containing an ALPACA_SECRET_KEY in the local directory.")
    api=REST(key_id=alpaca_api_key,secret_key=secret_key)
    df=api.get_bars("SOYB", TimeFrame.Day, start_date, end_date, adjustment='raw').df
    return(df)

def make_features_targets(dataframe,close=True,volume=False,trade_count=False,vwap=False):
    if close == True:
        dataframe['previous_close']=dataframe['close'].shift(1)
    if volume == True:
        dataframe['previous_volume']=dataframe['volume'].shift(1)
    if trade_count == True:
        dataframe['previous_trade_count']=dataframe['trade_count'].shift(1)
    if vwap == True:
        dataframe['previous_vwap']=dataframe['vwap'].shift(1)
    dataframe.dropna(inplace=True)
    X=dataframe[dataframe.columns[dataframe.columns.str.contains('previous')]]
    y=dataframe[['close']]
    return(X,y)

def cl_make_features_targets(dataframe,close=True,volume=False,trade_count=False,vwap=False):
    dataframe['target'] = np.where(dataframe['open'] < dataframe['close'], 1, 0)
    if close == True:
        dataframe['previous_close']=dataframe['close'].shift(1)
    if volume == True:
        dataframe['previous_volume']=dataframe['volume'].shift(1)
    if trade_count == True:
        dataframe['previous_trade_count']=dataframe['trade_count'].shift(1)
    if vwap == True:
        dataframe['previous_vwap']=dataframe['vwap'].shift(1)
    dataframe.dropna(inplace=True)
    X=dataframe[dataframe.columns[dataframe.columns.str.contains('previous')]]
    y=dataframe[['target']]
    return(X,y)

def train_test_split_by_date(X,y,division_factor):
    if len(X) != len(y):
        raise Exception("The length of the training and testing features is not the same")
    training_end_date=int(len(X)//division_factor)
    X_train=X.iloc[:training_end_date,:]
    y_train=y.iloc[:training_end_date,:]
    X_test=X.iloc[training_end_date:,:]
    y_test=y.iloc[training_end_date:,:]
    return(X_train,X_test,y_train,y_test)

def SVM_regressor(X_train,X_test,y_train,y_test):
    scaler=StandardScaler()
    scaler.fit(X_train)
    X_train_scaled=scaler.transform(X_train)
    X_test_scaled=scaler.transform(X_test)
    model=SVR()
    model.fit(X=X_train_scaled,y=y_train['close'])
    predicts=model.predict(X_test_scaled)
    predicts_df=pd.DataFrame({'close':y_test['close'],'predicted_close':predicts})
    return(predicts_df)

def SVM_classifier(X_train,X_test,y_train,y_test):
    scaler=StandardScaler()
    scaler.fit(X_train)
    X_train_scaled=scaler.transform(X_train)
    X_test_scaled=scaler.transform(X_test)
    model=SVC()
    model.fit(X=X_train_scaled,y=y_train['target'])
    predicts=model.predict(X_test_scaled)
    predicts_df=pd.DataFrame({'signal':y_test['target'],'predicted signal':predicts})
    return(predicts_df)

def classify_svm(df,division_factor,close=True,volume=True,trade_count=True,vwap=True):
    X,y=cl_make_features_targets(df,close=close,volume=volume,trade_count=trade_count,vwap=vwap)
    X_train,X_test,y_train,y_test=train_test_split_by_date(X,y,division_factor=division_factor)
    predictions_df=SVM_classifier(X_train,X_test,y_train,y_test)
    return(predictions_df)

def regressor_svm(df,division_factor,close=True,volume=True,trade_count=True,vwap=True):
    X,y=make_features_targets(df,close=close,volume=volume,trade_count=trade_count,vwap=vwap)
    X_train,X_test,y_train,y_test=train_test_split_by_date(X,y,division_factor=division_factor)
    predictions_df=SVM_regressor(X_train,X_test,y_train,y_test)
    return(predictions_df)

def r2(y_actual,y_predicted):
    ybar=(sum(y_actual))/len(y_actual)
    r_squared=1-(sum((y_actual-y_predicted)**2)/sum((y_actual-ybar)**2))
    return(r_squared)