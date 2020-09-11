from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
import pickle
import pandas as pd
import numpy as np
from sklearn import preprocessing
import yfinance as yf

# Define Functions
def chrono_var(days,df,close_column,low_column,high_column,minimum,maximum):
    df = df.reset_index()
    for i in range((len(df.index)-days)):
        for d in range(1,(days+1)):
            i2 = i + d
            current_close = df.loc[i, close_column]
            future_low = df.loc[i2, low_column]
            future_high = df.loc[i2, high_column]
            var_low =  (future_low/current_close)-1
            var_high = (future_high/current_close)-1
            if var_low<=minimum:
                df.loc[i,'Target(x)'] = 'Not'
                break
            elif var_high>=maximum:
                df.loc[i,'Target(x)'] = 'Buy'
                break
            else:
                df.loc[i,'Target(x)'] = 'Not'
                continue
    return df

first_day = "2018-01-01"
last_day = "2020-09-11"

SPY = yf.download("SPY", start=first_day, end=last_day,actions=False)
AMD = yf.download("AMD", start=first_day, end=last_day,actions=False)

df_AMD = pd.DataFrame(AMD)
df_SPY = pd.DataFrame(SPY)

del df_AMD["Close"]
del df_SPY["Close"]

df_Total = pd.merge(df_AMD, df_SPY, on="Date", how="inner")

df_Total.rename(columns={'Open_x':'Open',
                          'High_x':'High',
                          'Low_x':'Low',
                          'Adj Close_x':'Close',
                          'Volume_x':'Volume',
                          'Open_y':'Open_SPY',
                          'High_y':'High_SPY',
                          'Low_y':'Low_SPY',
                          'Adj Close_y':'Close_SPY',
                          'Volume_y':'Volume_SPY'},
                          inplace=True)

# Rolling AVG AMD
df_Total["MA(5)"] = round(df_Total["Close"].rolling(5).mean(),2)
df_Total["MA(10)"] = round(df_Total["Close"].rolling(10).mean(),2)
df_Total["MA(50)"] = round(df_Total["Close"].rolling(50).mean(),2)
df_Total["MA(100)"] = round(df_Total["Close"].rolling(100).mean(),2)
df_Total["MA(150)"] = round(df_Total["Close"].rolling(150).mean(),2)
df_Total["MA(200)"] = round(df_Total["Close"].rolling(200).mean(),2)

# Rolling AVG SPY
df_Total["MA(5)_SPY"] = round(df_Total["Close_SPY"].rolling(5).mean(),2)
df_Total["MA(10)_SPY"] = round(df_Total["Close_SPY"].rolling(10).mean(),2)
df_Total["MA(50)_SPY"] = round(df_Total["Close_SPY"].rolling(50).mean(),2)
df_Total["MA(100)_SPY"] = round(df_Total["Close_SPY"].rolling(100).mean(),2)
df_Total["MA(150)_SPY"] = round(df_Total["Close_SPY"].rolling(150).mean(),2)
df_Total["MA(200)_SPY"] = round(df_Total["Close_SPY"].rolling(200).mean(),2)

# Create Label/Target variable
df_Total = chrono_var(5,df_Total,"Close","Low","High",-0.02,0.1)

# Relative Volume last 10 days
df_Total["Rel. Vol(10)"] = round(df_Total["Volume"]/(df_Total["Volume"].rolling(10).mean())-1,2)
df_Total["Rel. Vol(10)_SPY"] =  round(df_Total["Volume_SPY"]/(df_Total["Volume_SPY"].rolling(10).mean())-1,2)

## RSI
window_length = 14
# Change in price
delta = df_Total['Close'].diff()
delta_SPY = df_Total['Close_SPY'].diff()
# Gain & Loss Series
up, down = delta.copy(), delta.copy()
up[up < 0] = 0
down[down > 0] = 0
up_SPY, down_SPY = delta_SPY.copy(), delta_SPY.copy()
up_SPY[up_SPY < 0] = 0
down_SPY[down_SPY > 0] = 0
# Calculate SMA
roll_up = up.rolling(window_length).mean()
roll_down = down.abs().rolling(window_length).mean()
roll_up_SPY = up_SPY.rolling(window_length).mean()
roll_down_SPY = down_SPY.abs().rolling(window_length).mean()
# Calculate the RSI based on SMA
RS = roll_up / roll_down
RSI = 100.0 - (100.0 / (1.0 + RS))
RS_SPY = roll_up_SPY / roll_down_SPY
RSI_SPY = 100.0 - (100.0 / (1.0 + RS_SPY))
# Merge to Main Dataframe
df_Total['RSI'] = RSI
df_Total['RSI_SPY'] = RSI_SPY

# Create relative MA Columns
df_Total['5>10'] = np.where(df_Total["MA(5)"]>df_Total["MA(10)"], "1", "0")
df_Total['10>50'] = np.where(df_Total["MA(10)"]>df_Total["MA(50)"], "1", "0")
df_Total['50>100'] = np.where(df_Total["MA(50)"]>df_Total["MA(100)"], "1", "0")
df_Total['100>150'] = np.where(df_Total["MA(100)"]>df_Total["MA(150)"], "1", "0")
df_Total['150>200'] = np.where(df_Total["MA(150)"]>df_Total["MA(200)"], "1", "0")

df_Total['5>10_SPY'] = np.where(df_Total["MA(5)_SPY"]>df_Total["MA(10)_SPY"], "1", "0")
df_Total['10>50_SPY'] = np.where(df_Total["MA(10)_SPY"]>df_Total["MA(50)_SPY"], "1", "0")
df_Total['50>100_SPY'] = np.where(df_Total["MA(50)_SPY"]>df_Total["MA(100)_SPY"], "1", "0")
df_Total['100>150_SPY'] = np.where(df_Total["MA(100)_SPY"]>df_Total["MA(150)_SPY"], "1", "0")
df_Total['150>200_SPY'] = np.where(df_Total["MA(150)_SPY"]>df_Total["MA(200)_SPY"], "1", "0")
# Remove NaN & Clean columns
df_Total = df_Total.dropna()
data=df_Total.loc[:,['Rel. Vol(10)',
       'Rel. Vol(10)_SPY', 'RSI', 'RSI_SPY', '5>10', '10>50', '50>100',
       '100>150', '150>200', '5>10_SPY', '10>50_SPY', '50>100_SPY',
       '100>150_SPY', '150>200_SPY']]

#  Convert objects to int
cols=['5>10', '10>50', '50>100', '100>150', '150>200', '5>10_SPY', '10>50_SPY', '50>100_SPY', '100>150_SPY', '150>200_SPY']
data[cols] = data[cols].apply(pd.to_numeric)

## Iterate over each model results
# Model list (imilar to other script)
model_list = {'Logistic','Naive_Bayes','Random_Forest','Knn','Neural_Net'}
# Iterations
for i in model_list:
    model = pickle.load(open('C:/Users/jrpgo/OneDrive - Rigor Consultoria e Gestão, SA/Pessoal/Python/Stocks/'+ str(i) + '.sav', 'rb'))
    prediction_perce = model.predict_proba(data)[:,0]
    df_Total[str(i)]=prediction_perce

# Export results
df_Total.to_excel(r'C:/Users/jrpgo/Desktop/Full.xlsx')
