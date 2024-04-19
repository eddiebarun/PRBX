from numpy import std
import pandas as pd

def met_year(sp500):
    sp500["Year"] = sp500["Date"].dt.year  # Extract the year as a new column
    return sp500

def met_month(sp500):
    sp500["Month"] = sp500["Date"].dt.month  # Extract the month as a new column
    return sp500

def met_rsi(sp500):
    #rsi or relative strength index is a a way of estimating if a stock is over/undersold, rsi over 70 = overbought, under 30 = oversold.
        
    n = 14 #number of days for rsi

    up_df, down_df = sp500[['Change in Price']].copy(), sp500[['Change in Price']].copy()

    # For up days, if the change is less than 0 set to 0.
    up_df.loc[(up_df['Change in Price'] < 0), 'Change in Price'] = 0

    # For down days, if the change is greater than 0 set to 0.
    down_df.loc[(down_df['Change in Price'] > 0), 'Change in Price'] = 0

    # We need change in price to be absolute.
    down_df['Change in Price'] = down_df['Change in Price'].abs()

    # Calculate the EWMA (Exponential Weighted Moving Average), meaning older values are given less weight compared to newer values.
    ewma_up = up_df.transform(lambda x: x.ewm(span = n).mean())
    ewma_down = down_df.transform(lambda x: x.ewm(span = n).mean())

    # Calculate the Relative Strength
    relative_strength = ewma_up / ewma_down

    # Calculate the Relative Strength Index
    relative_strength_index = 100.0 - (100.0 / (1.0 + relative_strength))

    # Add the info to the data frame.
    sp500['down_days'] = down_df['Change in Price']
    sp500['up_days'] = up_df['Change in Price']
    sp500['RSI'] = relative_strength_index
    
   
    
    return sp500
        
def met_sto_osc(sp500):
    # Calculate the Stochastic Oscillator
    n = 14

    # Make a copy of the high and low column.
    low_14 = sp500['Low'].copy()
    high_14 = sp500['High'].copy()

    # Group by symbol, then apply the rolling function and grab the Min and Max.
    low_14 = low_14.transform(lambda x: x.rolling(window = n).min())
    high_14 = high_14.transform(lambda x: x.rolling(window = n).max())

    # Calculate the Stochastic Oscillator.
    k_percent = 100 * ((sp500['Close'] - low_14) / (high_14 - low_14))

    # Add the info to the data frame.
    sp500['Low_14'] = low_14
    sp500['High_14'] = high_14
    sp500['K_percent'] = k_percent

    return sp500

def met_macd(sp500):
    #Calculate the MACD and signal line indicators
    #Calculate the short term exponential moving average (EMA)
    shortEMA = sp500['Close'].ewm(span=12,adjust=False).mean()

    #Calculate the long term exponential moving average (EMA)
    longEMA = sp500['Close'].ewm(span=26,adjust=False).mean()

    #Calculate the MACD line
    MACD = shortEMA - longEMA

    #calculate the signal line
    signal = MACD.ewm(span=9, adjust=False).mean()

    #set results back  to main dataframe
    sp500['MACD'] = MACD
    sp500['Signal Line'] = signal

    return sp500

def met_wil_per(sp500):

    #define period for calculation
    n = 14

    #calculate high low
    low_14 = sp500['Low'].copy()
    high_14 = sp500['High'].copy()

    low_14 = low_14.transform(lambda x: x.rolling(window = n).min())
    high_14 = high_14.transform(lambda x: x.rolling(window = n).max())

    #calculate william %R
    r_percent = ((high_14 - sp500['Close']) / (high_14 - low_14)) * -100

    # add the info to the data frame
    sp500['Will Percent'] = r_percent

    return sp500

def met_prc(sp500):
       
    sp500['Price Rate Change'] = sp500['Close'].transform(lambda x: x.pct_change(periods = 9))

    return sp500

def met_obv(sp500):
    #calculate on balance volume
    volume = sp500['Volume'].copy()
    change = sp500['Close'].diff().copy()

    prev_obv = 0
    obv_values = []

    for i,j in zip(change,volume):

        if i > 0:
            current_obv = prev_obv + j
        elif i < 0:
            current_obv = prev_obv - j
        else:
            current_obv = prev_obv

        prev_obv = current_obv
        obv_values.append(current_obv)

    obv_values = pd.Series(obv_values, index = sp500.index)

    #add it to the data frame
    sp500['On Balance Volume'] = obv_values

    return sp500

def met_bbands(sp500):
    std = 2
    n = 14
    
    sp500["MA"] = sp500["Close"].rolling(window=n).mean()
    sp500["Upper Band"] = sp500["MA"] + std * sp500["Close"].rolling(window=n).std()
    sp500["Lower Band"] = sp500["MA"] - std * sp500["Close"].rolling(window=n).std()
    
    return sp500

def all_metrics(sp500):
    sp500 = met_year(sp500)
    sp500 = met_month(sp500)
    sp500 = met_rsi(sp500)
    sp500 = met_sto_osc(sp500)
    sp500 = met_macd(sp500)
    sp500 = met_wil_per(sp500)
    sp500 = met_prc(sp500)
    sp500 = met_obv(sp500)
    sp500 = met_bbands(sp500)
    
    return sp500
