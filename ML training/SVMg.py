import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
import seaborn as sn
from sklearn.metrics import classification_report
df = pd.read_csv('Train.csv')
x_train = df.drop(columns=['class'])
y_train= df['class']
y_trainsample = y_train.head(100)
for column in x_train.columns:
    if x_train[column].dtype==type(object):
         labelencoder = LabelEncoder()
         x_train[column] = labelencoder.fit_transform(x_train[column])
x_trains = x_train.head(100)
labelencoder = LabelEncoder()

y_train = labelencoder.fit_transform(y_trainsample)
model = SVC(kernel='rbf', probability=True, C=1, gamma ='scale')
model.fit(x_trains, y_train)
df2 = pd.read_csv('Test.csv')
x1 = df2.drop(columns = ['class'])
y1= df2['class']
y2 = y1.head(100)
for column in x1.columns:
    if x1[column].dtype == type(object):
        labelencoder1 = LabelEncoder()
        x1[column] = labelencoder1.fit_transform(x1[column])
labelencoder = LabelEncoder()

y_test = labelencoder.fit_transform(y2)
x_test = x1.head(100)
# Predict class labels on training data
pred_labels_tr = model.predict(x_trains)
    # Predict class labels on a test data
pred_labels_te = model.predict(x_test)
print('----- Evaluation on Training Data -----')
score_tr = model.score(x_trains, y_train)
print('Accuracy Score: ', score_tr)
# Look at classification report to evaluate the model
print(classification_report(y_train, pred_labels_tr))
print('--------------------------------------------------------')
# Use score method to get accuracy of the model
print('----- Evaluation on Test Data -----')
score_te = model.score(x_test, y_test)
print('Accuracy Score: ', score_te)
# Look at classification report to evaluate the model
print(classification_report(y_test, pred_labels_te))
print('--------------------------------------------------------')
cm = pd.crosstab(y_test, pred_labels_te, rownames=['Actual'], colnames=['Predicted'], margins = True)
sn.heatmap(cm, annot=True)
plt.show()
tp=tn=fp=fn=0

for i in range(len(y_test)):
    if y_test[i]==0:
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
