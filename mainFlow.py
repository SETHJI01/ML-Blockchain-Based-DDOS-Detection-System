import pickle
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from joblib import dump, load
import time
import subprocess
from transactions import addData, checkData, getIP


df = pd.read_csv("data2.csv")
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.fillna(9999999, inplace=True)


def checkValidIP(ip):
    # body
    return True


timeThreshold = 1234  # to be decided later


def timeFormat(st):
    stri = st[6:10] + st[3:5] + st[0:2] + st[11:5]
    strin = st[11:13] + st[14:16] + st[17:19]
    typ = st[20:]
    if typ == "PM":
        it = int(strin)
        it = it + 120000
        strin = str(it)
    return int(stri + strin)


def currentTimeFormat():
    obj = time.gmtime()
    st = (
        str(obj.tm_year)
        + str(obj.tm_mon)
        + str(obj.tm_mday)
        + str(obj.tm_hour)
        + str(obj.tm_min)
        + str(obj.tm_sec)
    )
    return int(st)


with open("rftree_model.pkl", "rb") as f:
    loaded_model = pickle.load(f)

for i in range(df.shape[0]):
    x = df.iloc[i : i + 1, np.r_[13, 16, 41, 42, 54, 72, 73, 75]]
    data = df.iloc[i : i + 1, np.r_[1, 2, 4, 7]].values
    current_src_ip = data[0][1]
    timestamp_ip = data[0][3]
    for column in x.columns:
        if x[column].dtype == type(object):
            st = "encoder" + str(column) + ".joblib"
            labelencoder = load(st)
            x[column] = labelencoder.fit_transform(x[column])

    if checkValidIP(current_src_ip):
        timeOfIpinBC = check(current_src_ip)

        # if ip is already there in BC
        if timeOfIpinBC:
            currTime = currentTimeFormat()
            timeOfCurrIp = timeFormat(timestamp_ip)
            # check the threshold for already blocked ip
            if timeOfCurrIp - timeOfIpinBC >= timeThreshold:
                # check the validity through ML model

                y_pre = loaded_model.predict(x)

                if y_pre == 1:  # ddos detected
                    # store  the IP in block chain
                    add(current_src_ip, timeOfCurrIp)

                    # block it from server command
                    # Build the command
                    # command = 'netsh advfirewall firewall add rule name="Block Source IP" dir=in action=block remoteip={}'.format(current_src_ip)

                    # Run the command
                    # subprocess.run(command, shell=True)

                else:
                    pass
            else:
                # Update the timestamp of the IP in block chain
                add(current_src_ip, timeOfCurrIp)

                # block it from server command
                # Build the command
                # command = 'netsh advfirewall firewall add rule name="Block Source IP" dir=in action=block remoteip={}'.format(current_src_ip)

                # Run the command
                # subprocess.run(command, shell=True)

        else:
            # iP is not there in block chain

            # check curr ip from Ml model

            y_pre = loaded_model.predict(x)

            if y_pre == 1:  # ddos detected
                # store  the IP in block chain
                add(current_src_ip, timeOfCurrIp)

                # block it from server command
                # Build the command
                # command = 'netsh advfirewall firewall add rule name="Block Source IP" dir=in action=block remoteip={}'.format(current_src_ip)

                # Run the command
                # subprocess.run(command, shell=True)

            else:
                pass
