import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
import seaborn as sn
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import classification_report
df = pd.read_csv('Train.csv')
x = df.drop(columns=['class'])
y= df['class']
for column in x.columns:
    if x[column].dtype==type(object):
         labelencoder = LabelEncoder()
         x[column] = labelencoder.fit_transform(x[column])
            
         
         
labelencoder = LabelEncoder()

y = labelencoder.fit_transform(y)
model = LinearRegression()
model.fit(x,y)
LinearRegression()
df2 = pd.read_csv('Test.csv')
x1 = df2.drop(columns = ['class'])
y1= df2['class']
for column in x1.columns:
    if x1[column].dtype == type(object):
        labelencoder1 = LabelEncoder()
        x1[column] = labelencoder1.fit_transform(x1[column])
labelencoder = LabelEncoder()

y1 = labelencoder.fit_transform(y1)
y_predict = model.predict(x1)
print(y1,y_predict)
tp=tn=fp=fn=0

for i in range(len(y1)):
    if y1[i]==0:
        if y_predict[i]<1:
            tp=tp+1
        else:
            fn=fn+1
    else:
        if y_predict[i]<1:
            fp=fp+1
        else:
            tn=tn+1

accuracy=(tp+tn)/(tp+tn+fp+fn)
precision=tp/(tp+fp)
recall=tp/(tp+fn)
f1=(2*precision*recall)/(precision+recall)

print("accuracy :",accuracy)
print("precission :",precision)
print("recall :",recall)
print("f1 score :",f1)
print("fpr :",fp/(fp+tn))
print("fnr :",fn/(tp+fn))


r_sq = model.score(x, y)
print('coefficient of determination:', r_sq)

# The mean squared error
print('Mean squared error: %.2f' % mean_squared_error(y1, y_predict))

# The coefficient of determination: 1 is perfect prediction
print('Coefficient of determination: %.2f' % r2_score(y1, y_predict))