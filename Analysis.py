import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report, precision_score, roc_auc_score, auc, roc_curve
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import resample
from sklearn import preprocessing
import yfinance as yf
import pickle
import itertools

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

first_day = "2000-01-01"
last_day = "2020-09-10"

SPY = yf.download("SPY", start=first_day, end=last_day,actions=False)
AMD = yf.download("AMD", start=first_day, end=last_day,actions=False)

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
df_Total = chrono_var(5,df_Total,"Close","Low","High",-0.02,0.05)

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
data=df_Total.loc[:,['Target(x)', 'Rel. Vol(10)',
       'Rel. Vol(10)_SPY', 'RSI', 'RSI_SPY', '5>10', '10>50', '50>100',
       '100>150', '150>200', '5>10_SPY', '10>50_SPY', '50>100_SPY',
       '100>150_SPY', '150>200_SPY']]

#  Convert objects to int
cols=['5>10', '10>50', '50>100', '100>150', '150>200', '5>10_SPY', '10>50_SPY', '50>100_SPY', '100>150_SPY', '150>200_SPY']
data[cols] = data[cols].apply(pd.to_numeric)


## Split Train Test ORDERED
train, test= np.split(data, [int(.70 *len(data))])
# Split test features and labels
X_test = test.drop(['Target(x)'],axis=1)
y_test = test['Target(x)']
## Balance Data
# separate minority and majority classes
train_majority = train[train['Target(x)']=="Not"]
train_minority = train[train['Target(x)']=="Buy"]
# upsample minority, 42 for reproducibility
minority_upsample = resample(train_minority,replace=True, n_samples=len(train_majority), random_state=42)
# combine majority and upsampled minority
train_balanced = pd.concat([train_majority, minority_upsample])
# Split train features and labels
X_train =  train_balanced.drop(['Target(x)'],axis=1)
y_train = train_balanced['Target(x)']

# ## Split Train Test UNORDERED
# # Divide Features & target
# X = data.drop(['Target(x)'],axis=1)
# y = data['Target(x)']
# # Create train and test data. 42 for reproducibility
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42,stratify=y)
# # Rejoin features and targets - train
# train = pd.concat([X_train, y_train], axis=1)
# train_majority = train[train['Target(x)']=="Not"]
# train_minority = train[train['Target(x)']=="Buy"]
# minority_upsample = resample(train_minority,replace=True, n_samples=len(train_majority), random_state=42)
# # combine majority and upsampled minority
# train_balanced = pd.concat([train_majority, minority_upsample])
# # Split train features and labels
# X_train =  train_balanced.drop(['Target(x)'],axis=1)
# y_train = train_balanced['Target(x)']

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
# Create Log ROC Curve Variables
y_pred_prob = GS_Log.predict_proba(X_test)[:,0]
fpr_log, tpr_log, thresholds_log = roc_curve(y_test, y_pred_prob, pos_label="Buy")
LogisticRegression_AUC = auc(fpr_log, tpr_log)


########################## Naive Bayes
# Create Steps & Pipeline
steps = [('scaler', StandardScaler()),
('naive', GaussianNB())]
pipeline_NB = Pipeline(steps)
# Hyperperameters
##No Hyperperameters to tune
# Fit and Evaluate
pipeline_NB.fit(X_train, y_train)
pipeline_NB.score(X_test,y_test)
y_pred = pipeline_NB.predict(X_test)
classification_report(y_test, y_pred)
# Create Log ROC Curve Variables
y_pred_prob = pipeline_NB.predict_proba(X_test)[:,0]
fpr_NB, tpr_NB, thresholds_NB = roc_curve(y_test, y_pred_prob, pos_label="Buy")
NaiveBayes_AUC = auc(fpr_NB, tpr_NB)

########################## Random Forest
steps = [('scaler', StandardScaler()),
('forest', RandomForestClassifier())]
pipeline = Pipeline(steps)
#Hyperperameters
criterion = ['gini', 'entropy']
max_depth = np.arange(5, 50, 2)
parameters = {'forest__criterion':criterion,'forest__max_depth':max_depth, 'forest__random_state':[1]}
#Fit and Evaluate
GS_Forest = GridSearchCV(pipeline,parameters)
GS_Forest.fit(X_train, y_train)
y_pred = GS_Forest.predict(X_test)
report = classification_report(y_test, y_pred)
# Create Forest ROC Curve Variables
y_pred_prob = GS_Forest.predict_proba(X_test)[:,0]
fpr_forest, tpr_forest, thresholds_forest = roc_curve(y_test, y_pred_prob, pos_label="Buy")
RandomForest_AUC = auc(fpr_forest, tpr_forest)

########################## KNN
steps = [('scaler', StandardScaler()),
('Knn', KNeighborsClassifier())]
pipeline = Pipeline(steps)
# Hyperperameters
weights = ['uniform', 'distance']
algorithm = ['ball_tree','kd_tree','brute']
n_neighbors = np.arange(5, 50, 2)
parameters = {'Knn__algorithm':algorithm,'Knn__weights':weights,'Knn__n_neighbors':n_neighbors}
# Fit and Evaluate
GS_Knn = GridSearchCV(pipeline,parameters)
GS_Knn.fit(X_train, y_train)
GS_Knn.score(X_test,y_test)
y_pred = GS_Knn.predict(X_test)
classification_report(y_test, y_pred)
# Create Knn ROC Curve Variables
y_pred_prob = GS_Knn.predict_proba(X_test)[:,0]
GS_Knn.predict_proba(X_test)
fpr_Knn, tpr_Knn, thresholds_Knn = roc_curve(y_test, y_pred_prob, pos_label="Buy")
Knn_AUC = auc(fpr_Knn, tpr_Knn)

########################## Neural Network
steps = [('scaler', StandardScaler()),
('MLP', MLPClassifier())]
pipeline = Pipeline(steps)
# Hyperperameters
hidden_layer_sizes = [(50,50,50), (50,100,50)]
solver = ['lbfgs']
alpha = [0.0001,0.001,0.01,0.1]
learning_rate = ['constant','adaptive']
parameters = {'MLP__hidden_layer_sizes':hidden_layer_sizes,'MLP__solver':solver,'MLP__alpha':alpha,'MLP__learning_rate':learning_rate,'MLP__random_state':[1],'MLP__max_iter':[10000]}
# Fit and Evaluate
GS_MLP = GridSearchCV(pipeline,parameters)
GS_MLP.fit(X_train, y_train)
GS_MLP.score(X_test, y_test)
y_pred = GS_MLP.predict(X_test)
classification_report(y_test, y_pred)
# Create Knn ROC Curve Variables
y_pred_prob = GS_MLP.predict_proba(X_test)[:,0]
GS_MLP.predict_proba(X_test)
fpr_MLP, tpr_MLP, thresholds_MLP = roc_curve(y_test, y_pred_prob, pos_label="Buy")
MLP_AUC = auc(fpr_MLP, tpr_MLP)

# Track AUC scores
AUC_scores =  [['Logistic Regression',LogisticRegression_AUC],
                ['Naive Bayes',NaiveBayes_AUC],
                ['Random Forest',RandomForest_AUC],
                ['Knn',Knn_AUC],
                ['MLP',MLP_AUC]]
scores_df = pd.DataFrame(AUC_scores,columns = ['Algorithm', 'AUC'])
scores_df
# Create ROC Plots
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_log, tpr_log, label='Logistic Regression')
plt.plot(fpr_NB, tpr_NB, label='Naive Bayes')
plt.plot(fpr_forest, tpr_forest, label='Random Forest')
plt.plot(fpr_Knn, tpr_Knn, label='KNN')
plt.plot(fpr_MLP, tpr_MLP, label='MLP')
plt.legend()
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves')

## Export All Models
# Model dictionary
model_dict = {'Logistic':GS_Log,'Naive_Bayes':pipeline_NB,'Random_Forest':GS_Forest,'Knn':GS_Knn,'Neural_Net':GS_MLP}
# Iterate over each model
for key in model_dict:
    filename = 'C:/Users/jrpgo/OneDrive - Rigor Consultoria e Gestão, SA/Pessoal/Python/Stocks/' + str(key) +'.sav'
    pickle.dump(model_dict[key], open(filename, 'wb'))
