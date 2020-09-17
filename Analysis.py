import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV,PredefinedSplit
from sklearn.metrics import confusion_matrix, classification_report, precision_score, roc_auc_score, auc, roc_curve
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.utils import resample
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA
from sklearn import preprocessing
import dask_ml.model_selection as dcv
import yfinance as yf
import pickle
import itertools

## Define Functions
# Create target (check if target value occures before acceptable drop)
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
# Shift columns X times
def multi_shift(df,columns,shift):
    for i in columns:
        if shift>1:
            y=1
        if shift<1:
            y=-1
        for t in range(1,(shift+y)):
            df.loc[:,str(i)+"_("+str(t) +")"]=df[str(i)].shift(t)
    return df

first_day = "2010-01-01"
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
df_Total = chrono_var(5,df_Total,"Close","Low","High",-0.02,0.10)

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
df_Total['5>10'] = np.where(df_Total["MA(5)"]>df_Total["MA(10)"], 1, 0)
df_Total['10>50'] = np.where(df_Total["MA(10)"]>df_Total["MA(50)"], 1, 0)
df_Total['50>100'] = np.where(df_Total["MA(50)"]>df_Total["MA(100)"], 1, 0)
df_Total['100>150'] = np.where(df_Total["MA(100)"]>df_Total["MA(150)"], 1, 0)
df_Total['150>200'] = np.where(df_Total["MA(150)"]>df_Total["MA(200)"], 1, 0)

df_Total['5>10_SPY'] = np.where(df_Total["MA(5)_SPY"]>df_Total["MA(10)_SPY"], 1, 0)
df_Total['10>50_SPY'] = np.where(df_Total["MA(10)_SPY"]>df_Total["MA(50)_SPY"], 1, 0)
df_Total['50>100_SPY'] = np.where(df_Total["MA(50)_SPY"]>df_Total["MA(100)_SPY"], 1, 0)
df_Total['100>150_SPY'] = np.where(df_Total["MA(100)_SPY"]>df_Total["MA(150)_SPY"], 1, 0)
df_Total['150>200_SPY'] = np.where(df_Total["MA(150)_SPY"]>df_Total["MA(200)_SPY"], 1, 0)

# Low, High, Open Relative Position to Close
df_Total['Low Pos'] = round(df_Total['Low']/df_Total['Close']-1,2)
df_Total['High Pos'] = round(df_Total['High']/df_Total['Close']-1,2)
df_Total['Open Pos'] = round(df_Total['Open']/df_Total['Close']-1,2)

df_Total['Low Pos_SPY'] = round(df_Total['Low_SPY']/df_Total['Close_SPY']-1,2)
df_Total['High Pos_SPY'] = round(df_Total['High_SPY']/df_Total['Close_SPY']-1,2)
df_Total['Open Pos_SPY'] = round(df_Total['Open_SPY']/df_Total['Close_SPY']-1,2)

# Apply time delay & Remove NaN & Clean columns
data=df_Total.loc[:,['MA(5)', 'MA(10)',
       'MA(50)', 'MA(100)', 'MA(150)', 'MA(200)', 'MA(5)_SPY', 'MA(10)_SPY',
       'MA(50)_SPY', 'MA(100)_SPY', 'MA(150)_SPY', 'MA(200)_SPY', 'Target(x)',
       'Rel. Vol(10)', 'Rel. Vol(10)_SPY', 'RSI', 'RSI_SPY', '5>10', '10>50',
       '50>100', '100>150', '150>200', '5>10_SPY', '10>50_SPY', '50>100_SPY',
       '100>150_SPY', '150>200_SPY', 'Low Pos', 'High Pos', 'Open Pos',
       'Low Pos_SPY', 'High Pos_SPY', 'Open Pos_SPY']]
data = multi_shift(data,data.columns[data.columns!="Target(x)"],5)
data = data.dropna()

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

# Force Split in GridSearch
split_index = [-1]*int(len(X_train)*0.7)+[0]*(len(X_train)-int(len(X_train)*0.7))
pds = PredefinedSplit(test_fold = split_index)

########################## Logistic Regression
# Create Steps & Pipeline
steps = [('scaler', StandardScaler()),
("PCA",PCA()),
('logistic', LogisticRegression())]
pipeline = Pipeline(steps)
# Hyperperameters
n_components = [20,40,80,120,190]
penalty = ['none']
solver = ['lbfgs']
parameters = {"PCA__n_components":n_components,'logistic__penalty':penalty,'logistic__solver':solver,'logistic__max_iter':[1000000000]}
# Fit and Evaluate
GS_Log = dcv.GridSearchCV(pipeline,parameters,scheduler='threading',error_score=0.0,scoring='roc_auc',cv=pds)
GS_Log.fit(X_train, y_train)
GS_Log.score(X_test,y_test)
y_pred = GS_Log.predict(X_test)
Logistic_report = classification_report(y_test, y_pred)
# Create Log ROC Curve Variables
y_pred_prob = GS_Log.predict_proba(X_test)[:,0]
fpr_log, tpr_log, thresholds_log = roc_curve(y_test, y_pred_prob, pos_label="Buy")
LogisticRegression_AUC = auc(fpr_log, tpr_log)

########################## Bagging
steps = [('scaler', StandardScaler()),
("PCA",PCA()),
('Bag', BaggingClassifier())]
pipeline = Pipeline(steps)
# Hyperperameters
n_components = [20,40,80,120,190]
base_estimator = [DecisionTreeClassifier()]
n_estimators=[5,10,15,20,50,100]
parameters = {"PCA__n_components":n_components,"Bag__base_estimator":base_estimator,"Bag__n_estimators":n_estimators}
# Fit and Evaluate
GS_Bag = dcv.GridSearchCV(pipeline,parameters,scheduler='threading',error_score=0.0,scoring='roc_auc',cv=pds)
GS_Bag.fit(X_train, y_train)
GS_Bag.score(X_test,y_test)
y_pred = GS_Bag.predict(X_test)
Logistic_report = classification_report(y_test, y_pred)
# Create Log ROC Curve Variables
y_pred_prob = GS_Bag.predict_proba(X_test)[:,0]
fpr_log, tpr_log, thresholds_log = roc_curve(y_test, y_pred_prob, pos_label="Buy")
Bagging_AUC = auc(fpr_log, tpr_log)

########################## Random Forest
steps = [('scaler', StandardScaler()),
("PCA",PCA()),
('forest', RandomForestClassifier())]
pipeline = Pipeline(steps)
#Hyperperameters
n_components = [20,40,80,120,190]
criterion = ['gini', 'entropy']
max_depth = np.arange(5, 50, 2)
parameters = {"PCA__n_components":n_components,'forest__criterion':criterion,'forest__max_depth':max_depth, 'forest__random_state':[1]}
#Fit and Evaluate
GS_Forest = dcv.GridSearchCV(pipeline,parameters,scheduler='threading',cv=pds)
GS_Forest.fit(X_train, y_train)
y_pred = GS_Forest.predict(X_test)
RandomForest_report =  classification_report(y_test, y_pred)
# Create Forest ROC Curve Variables
y_pred_prob = GS_Forest.predict_proba(X_test)[:,0]
fpr_forest, tpr_forest, thresholds_forest = roc_curve(y_test, y_pred_prob, pos_label="Buy")
RandomForest_AUC = auc(fpr_forest, tpr_forest)

########################## KNN
steps = [('scaler', StandardScaler()),
("PCA",PCA()),
('Knn', KNeighborsClassifier())]
pipeline = Pipeline(steps)
# Hyperperameters
n_components = [20,40,80,120,190]
weights = ['uniform', 'distance']
algorithm = ['ball_tree','kd_tree','brute']
n_neighbors = np.arange(5, 50, 2)
parameters = {"PCA__n_components":n_components,'Knn__algorithm':algorithm,'Knn__weights':weights,'Knn__n_neighbors':n_neighbors}
# Fit and Evaluate
GS_Knn = dcv.GridSearchCV(pipeline,parameters,scheduler='threading',cv=pds)
GS_Knn.fit(X_train, y_train)
GS_Knn.score(X_test,y_test)
y_pred = GS_Knn.predict(X_test)
Knn_report = classification_report(y_test, y_pred)
# Create Knn ROC Curve Variables
y_pred_prob = GS_Knn.predict_proba(X_test)[:,0]
GS_Knn.predict_proba(X_test)
fpr_Knn, tpr_Knn, thresholds_Knn = roc_curve(y_test, y_pred_prob, pos_label="Buy")
Knn_AUC = auc(fpr_Knn, tpr_Knn)

########################## Neural Network
steps = [('scaler', StandardScaler()),
("PCA",PCA()),
('MLP', MLPClassifier())]
pipeline = Pipeline(steps)
# Hyperperameters
n_components = [20,40,80,120,190]
hidden_layer_sizes = [(190,50,50), (190,100,50),(190,50,2), (190,100,2),(190,50,1), (190,100,1)]
solver = ['lbfgs','sgd']
alpha = [0.00001,0.0001,0.001,0.01,0.1]
learning_rate = ['constant','adaptive']
parameters = {"PCA__n_components":n_components,'MLP__hidden_layer_sizes':hidden_layer_sizes,'MLP__solver':solver,'MLP__alpha':alpha,'MLP__learning_rate':learning_rate,'MLP__random_state':[42], 'MLP__max_iter':[30000]}
# Fit and Evaluate
GS_MLP = dcv.GridSearchCV(pipeline,parameters,scheduler='threading',cv=pds)
GS_MLP.fit(X_train, y_train)
GS_MLP.score(X_test, y_test)
y_pred = GS_MLP.predict(X_test)
MLP_report = classification_report(y_test, y_pred)
# Create Knn ROC Curve Variables
y_pred_prob = GS_MLP.predict_proba(X_test)[:,0]
GS_MLP.predict_proba(X_test)
fpr_MLP, tpr_MLP, thresholds_MLP = roc_curve(y_test, y_pred_prob, pos_label="Buy")
MLP_AUC = auc(fpr_MLP, tpr_MLP)

########################## SVM
steps = [('scaler', StandardScaler()),
("PCA",PCA()),
('SVM', SVC())]
pipeline = Pipeline(steps)
# Hyperperameters
n_components = [20,40,80,120,190]
C = [0.1, 1, 10, 100, 1000]
gamma = [1, 0.1, 0.01, 0.001, 0.0001]
kernel = ['linear', 'poly']
parameters = {"PCA__n_components":n_components,'SVM__C':C, 'SVM__gamma':gamma, 'SVM__kernel':kernel, 'SVM__random_state':[42],'SVM__probability':[True]}
# Fit and Evaluate
GS_SVM = dcv.RandomizedSearchCV(pipeline,parameters,scheduler='threading',random_state=42, n_iter=20,cv=pds)
GS_SVM.fit(X_train, y_train)
GS_SVM.score(X_test, y_test)
y_pred = GS_SVM.predict(X_test)
SVM_report =  classification_report(y_test, y_pred)
# Create Knn ROC Curve Variables
y_pred_prob = GS_SVM.predict_proba(X_test)[:,0]
GS_SVM.predict_proba(X_test)
fpr_SVM, tpr_SVM, thresholds_SVM = roc_curve(y_test, y_pred_prob, pos_label="Buy")
SVM_AUC = auc(fpr_SVM, tpr_SVM)


SVM_AUC=None
# Track AUC scores
AUC_scores =  [['Logistic Regression',LogisticRegression_AUC],
                ['Random Forest',RandomForest_AUC],
                ['Knn',Knn_AUC],
                ['MLP',MLP_AUC],
                ['SVM',SVM_AUC]]
scores_df = pd.DataFrame(AUC_scores,columns = ['Algorithm', 'AUC'])
scores_df
# Create ROC Plots
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_log, tpr_log, label='Logistic Regression')
plt.plot(fpr_forest, tpr_forest, label='Random Forest')
plt.plot(fpr_Knn, tpr_Knn, label='KNN')
plt.plot(fpr_MLP, tpr_MLP, label='MLP')
plt.plot(fpr_SVM, tpr_SVM, label='SVM')
plt.legend()
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves')

## Export All Models
# Model dictionary
model_dict = {'Logistic':GS_Log, 'Random_Forest':GS_Forest,'Knn':GS_Knn,'Neural_Net':GS_MLP}
# Iterate over each model
for key in model_dict:
    filename = 'C:/Users/jrpgo/OneDrive - Rigor Consultoria e Gestão, SA/Pessoal/Python/Stocks/' + str(key) +'.sav'
    pickle.dump(model_dict[key], open(filename, 'wb'))
