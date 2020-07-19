import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

pd.options.display.html.table_schema = True
pd.options.display.max_rows = 100

data_AMD = pd.read_csv("C:/Users/jrpgo/OneDrive - Rigor Consultoria e Gestão, SA/Pessoal/Python/Stocks/Data/AMD.csv", parse_dates =["Date"])
data_SPY = pd.read_csv("C:/Users/jrpgo/OneDrive - Rigor Consultoria e Gestão, SA/Pessoal/Python/Stocks/Data/SPY.csv", parse_dates =["Date"])

df_AMD = pd.DataFrame(data_AMD)
df_SPY = pd.DataFrame(data_SPY)

del df_AMD["Adj Close"]
del df_SPY["Adj Close"]

df_Total = pd.merge(df_AMD, df_SPY, on="Date", how="inner")

df_Total.rename(columns={'Open_x':'Open',
                          'High_x':'High',
                          'Low_x':'Low',
                          'Close_x':'Close',
                          'Volume_x':'Volume',
                          'Open_y':'Open_SPY',
                          'High_y':'High_SPY',
                          'Low_y':'Low_SPY',
                          'Close_y':'Close_SPY',
                          'Volume_y':'Volume_SPY'},
                          inplace=True)

# Rolling AVG AMD
df_Total["MA(5)"] = df_Total["Close"].rolling(5).mean()
df_Total["MA(10)"] = df_Total["Close"].rolling(10).mean()
df_Total["MA(50)"] = df_Total["Close"].rolling(50).mean()
df_Total["MA(100)"] = df_Total["Close"].rolling(100).mean()
df_Total["MA(150)"] = df_Total["Close"].rolling(150).mean()
df_Total["MA(200)"] = df_Total["Close"].rolling(200).mean()

# Rolling AVG SPY
df_Total["MA(5) SPY"] = df_Total["Close_SPY"].rolling(5).mean()
df_Total["MA(10) SPY"] = df_Total["Close_SPY"].rolling(10).mean()
df_Total["MA(50) SPY"] = df_Total["Close_SPY"].rolling(50).mean()
df_Total["MA(100) SPY"] = df_Total["Close_SPY"].rolling(100).mean()
df_Total["MA(150) SPY"] = df_Total["Close_SPY"].rolling(150).mean()
df_Total["MA(200) SPY"] = df_Total["Close_SPY"].rolling(200).mean()

# How much Growth % next 5 days
df_Total["Max_High_Next(5)%"] = (df_Total["High"].rolling(5).max().shift(-5)/df_Total["Close"])-1

# Did Grown more than 5% next 5 days?
df_Total["Max_High_Next(5)?"] = np.where(df_Total["Max_High_Next(5)%"]>=0.05, "Y", "N")

# Relative Volume last 10 days
df_Total["Rel. Vol (10)"] = df_Total["Volume"]/(df_Total["Volume"].rolling(10).mean())-1
df_Total["Rel. Vol (10) SPY"] =  df_Total["Volume_SPY"]/(df_Total["Volume_SPY"].rolling(10).mean())-1

# Date as Index
df_Total.set_index('Date', inplace=True)

# RSI
RSI = df_Total.loc[:,["Close","Close_SPY"]]
RSI["Close_Change"] = RSI["Close"]-RSI["Close"].shift(1)
RSI["Close_SPY_Change"] = RSI["Close_SPY"]-RSI["Close_SPY"].shift(1)

RSI["Gain"] = np.where(RSI["Close_Change"] > 0, RSI["Close_Change"], np.nan)
RSI["Gain_SPY"] = np.where(RSI["Close_SPY_Change"] > 0, RSI["Close_SPY_Change"], np.nan)

RSI["Loss"] = np.where(RSI["Close_Change"] < 0, RSI["Close_Change"], np.nan)
RSI["Loss_SPY"] = np.where(RSI["Close_SPY_Change"] < 0, RSI["Close_SPY_Change"], np.nan)

RSI.head(35)
