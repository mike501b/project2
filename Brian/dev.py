from os import getenv
from dotenv import load_dotenv
from alpaca_trade_api.rest import REST, TimeFrame, TimeFrameUnit
import pandas as pd
from datetime import date, timedelta

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
