import numpy as np
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics
import sklearn
from sklearn.metrics import classification_report
df = pd.read_csv('TRAIN.csv')
x = df.drop(columns = ['class'])
y = df['class']
for column in x.columns:
    if x[column].dtype == type(object):
        labelencoder = LabelEncoder()
        x[column] = labelencoder.fit_transform(x[column])
        
labelencoder = LabelEncoder()

y = labelencoder.fit_transform(y)
df2 = pd.read_csv('TEST.csv')
x1 = df2.drop(columns = ['class'])
for column in x1.columns:
    if x1[column].dtype == type(object):
        labelencoder1 = LabelEncoder()
        x1[column] = labelencoder1.fit_transform(x1[column])
y1 = df2['class']
labelencoder = LabelEncoder()
y1 = labelencoder.fit_transform(y1)
regressor = RandomForestRegressor(n_estimators=20, random_state=0)
regressor.fit(x, y)
y_pre = regressor.predict(x1)

y_pre = y_pre.astype(int)

print(classification_report(y1, y_pre))

# print('----- Evaluation on Training Data -----')
# score_tr = regressor.score(x, y)
# print('Accuracy Score: ', score_tr)
