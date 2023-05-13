import pickle
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from joblib import dump,load

df=pd.read_csv("data2.csv")
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.fillna(9999999, inplace=True)
x = df.iloc[25500:26000,np.r_[9,13,16,18,26,28,29,41,42,54,72,73,75]]
y = df.iloc[25500:26000,-1].values
xx=df.iloc[:,:8].values

with open("rftree_model.pkl",'rb') as f:    
    loaded_model = pickle.load(f)

for column in x.columns:
    if x[column].dtype == type(object):
        st = 'encoder'+str(column)+'.joblib'
        labelencoder = load(st)
        x[column] = labelencoder.fit_transform(x[column])
        print(column)
  
y_pre=loaded_model.predict(x)

y_pre = y_pre.astype(int)

x=x.values
for i in range(len(x)):
    print(x[i])

for i in range(len(y)):
    print(y[i],y_pre[i])


ll=[]
di={}


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

# print(dii)

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
