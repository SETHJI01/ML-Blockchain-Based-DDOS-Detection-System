import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
import seaborn as sn
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

df = pd.read_csv('datab3.csv')
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.fillna(9999999, inplace=True)

df1 = pd.read_csv('data3.csv')
df1.replace([np.inf, -np.inf], np.nan, inplace=True)
df1.fillna(9999999, inplace=True)
# print(df.isnull().values.any())
# df.fillna(0)
Xx = df.iloc[:,:-1]
Yy = df.iloc[:,-1]
print(Xx.shape , Yy.shape)
# x,  x1  , y, y1 = train_test_split(Xx,Yy ,random_state=104, test_size=0.25,shuffle=True)
# print(x,y)

x = df.iloc[:,np.r_[0,1,2,4,7,13,54,72,73,75]]
y = df.iloc[:,-1]

x1 = df1.iloc[:,np.r_[0,1,2,4,7,13,54,72,73,75]]
y1= df1.iloc[:,-1]

xxx=x.values
xx=x1.values
for column in x.columns:
    if x[column].dtype == type(object):
        labelencoder = LabelEncoder()
        x[column] = labelencoder.fit_transform(x[column])
        
labelencoder = LabelEncoder()

y = labelencoder.fit_transform(y)
for column in x1.columns:
    if x1[column].dtype == type(object):
        labelencoder1 = LabelEncoder()
        x1[column] = labelencoder1.fit_transform(x1[column])


labelencoder = LabelEncoder()
y1 = labelencoder.fit_transform(y1)
model = GaussianNB()
model.fit(x, y)

#x_test = x1.head(100)
# Predict class labels on training data
pred_labels_tr = model.predict(x)
    # Predict class labels on a test data
pred_labels_te = model.predict(x1)
print('----- Evaluation -----')
score_tr = model.score(x, y)
print('Accuracy Score: ', score_tr)
# Look at classification report to evaluate the model
print(classification_report(y1, pred_labels_te))
tp=tn=fp=fn=0

for i in range(len(y1)):
    if y1[i]==0:
        if pred_labels_te[i]==0:
            tp=tp+1
        else:
            fn=fn+1
    else:
        if pred_labels_te[i]==0:
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
