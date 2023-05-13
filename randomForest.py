from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics
import sklearn
from sklearn.metrics import classification_report
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split
import pickle
import matplotlib.pyplot as plt
from joblib import dump,load

df = pd.read_csv('datab2.csv')
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.fillna(9999999, inplace=True)
df1 = pd.read_csv('data2.csv')
df1.replace([np.inf, -np.inf], np.nan, inplace=True)
df1.fillna(9999999, inplace=True)
# print(df.isnull().values.any())
# df.fillna(0)
# top_f=[1,75,4,2,0,7,73,72,54,13] [ 1 75  4  2  0  7 73 72 54 13  9  6 16 41 42 18  3 26 29 28]


# selecting top 13 features for training
x = df.iloc[:,np.r_[9,13,16,18,26,28,29,41,42,54,72,73,75]]
y = df.iloc[:,-1]

xx = df1.iloc[:,:8].values

# selecting top 13 features for testing
x1 = df1.iloc[:,np.r_[9,13,16,18,26,28,29,41,42,54,72,73,75]]
y1= df1.iloc[:,-1]
# print(Xx.shape , Yy.shape)
# x,  x1  , y, y1 = train_test_split(Xx,Yy ,random_state=104, test_size=0.25,shuffle=True)
# print(x,y)

for column in x.columns:
    print(column)
    if x[column].dtype == type(object):
        labelencoder = LabelEncoder()
        x[column] = labelencoder.fit_transform(x[column])
        
labelencoder = LabelEncoder()
# print(y)
y = labelencoder.fit_transform(y)
# print(y[3])
for column in x1.columns:
    if x1[column].dtype == type(object):
        labelencoder = LabelEncoder()
        x1[column] = labelencoder.fit_transform(x1[column])
        st = 'encoder'+str(column)+'.joblib'
        dump(labelencoder,st)

labelencoder = LabelEncoder()
y1 = labelencoder.fit_transform(y1)

# rf=RandomForestClassifier(n_estimators=100,random_state=42)
# rf.fit(x,y)
# feature_importance = rf.feature_importances_
# plt.bar(range(len(feature_importance)), feature_importance)
# plt.title("Feature importances")
# plt.xlabel("Feature index")
# plt.ylabel("Importance")
# plt.show()

# k = 20
# top_k_features = np.argsort(feature_importance)[::-1][:k]

# print(top_k_features)

regressor = RandomForestRegressor(n_estimators=20, random_state=0)
regressor.fit(x, y)
y_pre = regressor.predict(x1)

y_pre = y_pre.astype(int)

print(classification_report(y1, y_pre))

tp=tn=fp=fn=0

for i in range(len(y1)):
    if y1[i]==0:
        if y_pre[i]==0:
            tp=tp+1
        else:
            fn=fn+1
    else:
        if y_pre[i]==0:
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


ll=[]
di={}

# for i,item in enumerate(y):
#         lll=[]
#         if item == 1 :
#             di[xxx[i][2]]=xxx[i][7]
#             ll.append(lll)

for i,item in enumerate(y_pre):
        lll=[]
        if item == 1 :
            di[xx[i][2]]=xx[i][7]
            ll.append(lll)


# print(di)
dii={}
cnt=1
ll=[]
for key in di:
    diii={}
    diii["IP"]=key
    diii["TimeStamp"]=di[key]
    ll.append(diii)
    dii["BlackList"+ str(cnt)]=diii
    cnt=cnt+1

print(dii)

def timeFormat(st) :
    stri =   st[6:10] + st[3:5] +st[0:2]  + st[11:5]
    strin = st[11:13] + st[14:16] + st[17:19]
    typ =  st[20:]
    if typ == "PM" :
        it = int(strin)
        it = it + 120000
        strin = str(it)
    return int(stri+strin)


for i in range(0,len(ll)) :
    st = ll[i]['TimeStamp']    
    timeInNum =   timeFormat(st) 
    ll[i]['TimeInNum']=timeInNum  


print(ll)
x1=x1.values
for i in range(3200,3241):
    print(x1[i])
# json_object = json.dumps(dii, indent = 9)
# print(json_object)

with open('rftree_model.pkl','wb') as f:
    pickle.dump(regressor,f)

# [{"IP": "172.31.69.25", "TimeStamp": "16/02/2018 11:23:51 PM"}, {"IP": "172.31.69.28", "TimeStamp": "22/02/2018 12:15:13 AM"}, {"IP": "18.218.229.235", "TimeStamp": "21/02/2018 11:46:28 PM"}, {"IP": "13.59.126.31", "TimeStamp": "16/02/2018 08:13:46 PM"}, {"IP": "18.219.193.20", "TimeStamp": "16/02/2018 11:24:04 PM"}, {"IP": "18.216.200.189", "TimeStamp": "20/02/2018 10:25:08"}, {"IP": "18.218.11.51", "TimeStamp": "22/02/2018 12:14:52 AM"}, {"IP": "52.14.136.135", "TimeStamp": "21/02/2018 11:57:37 PM"}, {"IP": "18.219.211.138", "TimeStamp": "15/02/2018 07:24:36 PM"}, {"IP": "18.219.9.1", "TimeStamp": "22/02/2018 12:28:23 AM"}, {"IP": "18.219.32.43", "TimeStamp": "21/02/2018 11:46:48 PM"}, {"IP": "18.219.5.43", "TimeStamp": "20/02/2018 10:46:16"}, {"IP": "18.218.55.126", "TimeStamp": "22/02/2018 12:06:29 AM"}, {"IP": "18.216.24.42", "TimeStamp": "21/02/2018 11:43:00 PM"}, {"IP": "192.168.2.109", "TimeStamp": "12/06/2010 10:20:36 PM"}, {"IP": "192.168.56.1", "TimeStamp": "12/06/2010 01:09:52 PM"}, {"IP": "18.218.115.60", "TimeStamp": "20/02/2018 10:32:44"}, {"IP": "192.168.56.102", "TimeStamp": "12/06/2010 11:59:11 AM"}, {"IP": "18.217.165.70", "TimeStamp": "15/02/2018 08:43:29 PM"}, {"IP": "192.168.3.117", "TimeStamp": "12/06/2010 05:20:36 PM"}, {"IP": "192.168.1.104", "TimeStamp": "13/06/2010 01:23:56 AM"}, {"IP": "192.168.2.108", "TimeStamp": "12/06/2010 06:56:24 PM"}, {"IP": "192.168.1.102", "TimeStamp": "12/06/2010 04:58:13 PM"}, {"IP": "192.168.2.112", "TimeStamp": "12/06/2010 02:20:45 PM"}, {"IP": "192.168.2.110", "TimeStamp": "13/06/2010 07:45:38 AM"}, {"IP": "192.168.1.101", "TimeStamp": "12/06/2010 02:23:26 PM"}, {"IP": "192.168.1.103", "TimeStamp": "12/06/2010 02:17:02 PM"}]