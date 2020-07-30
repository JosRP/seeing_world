import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
import yfinance as yf

SPY = yf.download("SPY", start="2018-07-22", end="2020-07-22",actions=False)
AMD = yf.download("AMD", start="2018-07-22", end="2020-07-22",actions=False)

# Need find way to remove -2% if befoire max 5%

pd.options.display.html.table_schema = False
pd.options.display.max_rows = 100

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

# Add day variable
days = 5

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

# Round Close_x
df_Total["Close"] = round(df_Total["Close"],2)
df_Total["Close_SPY"] = round(df_Total["Close_SPY"],2)

# How much Max and Min Growth % next 5 days
df_Total["Max(x)"] = round((df_Total["High"].rolling(days).max().shift(-5)/df_Total["Close"])-1,2)
#df_Total["Min(x)"] = round((df_Total["High"].rolling(days).min().shift(-5)/df_Total["Close"])-1,2)

# Did Grown more than 5% next 5 days?
df_Total["Target(x)"] = np.where(df_Total["Max(x)"]>=0.05, "Buy", "Not")

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

# Clean columns
data=df_Total.loc[:,['Target(x)', 'Max(x)', 'Rel. Vol(10)',
       'Rel. Vol(10)_SPY', 'RSI', 'RSI_SPY', '5>10', '10>50', '50>100',
       '100>150', '150>200', '5>10_SPY', '10>50_SPY', '50>100_SPY',
       '100>150_SPY', '150>200_SPY']]

# Remove NAN convert objects to int
data = data.dropna()
cols=['5>10', '10>50', '50>100', '100>150', '150>200', '5>10_SPY', '10>50_SPY', '50>100_SPY', '100>150_SPY', '150>200_SPY']
data[cols] = data[cols].apply(pd.to_numeric)

# EDA
sns.scatterplot(x='Rel. Vol(10)',y='Max(x)',data=data)
plt.close()
sns.scatterplot(x='RSI',y='Max(x)',data=data)
plt.close()
sns.boxplot(x='5>10',y='Max(x)',data=data)
plt.close()

# Divide Features & target
X = data.drop(['Target(x)','Max(x)'],axis=1).values
y = data['Target(x)'].values
# Create train and test data. 42 for reproducibility
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42,stratify=y)

########################## Decision Tree
# Create Steps & Pipeline
steps = [('scaler', StandardScaler()),
('decisiontree', tree.DecisionTreeClassifier())]
pipeline = Pipeline(steps)
#Hyperperameters
criterion = ['gini', 'entropy']
max_depth = np.linspace(1, 32, 32, endpoint=True)
parameters = {'decisiontree__criterion':criterion,'decisiontree__max_depth':max_depth,'decisiontree__random_state':[1]}
#Fit and Evaluate
GS_Tree = GridSearchCV(pipeline,parameters)
GS_Tree.fit(X_train, y_train)
GS_Tree.score(X_test,y_test)
y_pred = GS_Tree.predict(X_test)
classification_report(y_test, y_pred)
# Create Tree ROC Curve Variables
y_pred_prob = GS_Tree.predict_proba(X_test)[:,0]
GS_Tree.predict_proba(X_test)
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob, pos_label="Buy")
# Create Decision Tree ROC Plot
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr, label='Decision Tree')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Decision Tree ROC Curve')


########################## Logistic Regression
# Create Steps & Pipeline
steps = [('scaler', StandardScaler()),
('logistic', LogisticRegression())]
pipeline = Pipeline(steps)
# Hyperperameters
penalty = ['none']
parameters = {'logistic__penalty':penalty}
# Fit and Evaluate
GS_Log = GridSearchCV(pipeline,parameters)
GS_Log.fit(X_train, y_train)
GS_Log.score(X_test,y_test)
y_pred = GS_Log.predict(X_test)
classification_report(y_test, y_pred)
# Create LOg ROC Curve Variables
y_pred_prob = GS_Log.predict_proba(X_test)[:,0]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob, pos_label="Buy")
# Create Logistic Regression ROC Plot
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr, label='Logistic Regression')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Logistic Regression ROC Curve')
