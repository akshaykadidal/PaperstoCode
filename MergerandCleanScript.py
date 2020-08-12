# -*- coding: utf-8 -*-
"""
Created on Sat Aug  8 17:15:36 2020

@author: akshaya.p.kadidal
"""

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from glob import glob
import matplotlib.pyplot as plt
from tslearn.clustering import TimeSeriesKMeans, silhouette_score
from tslearn.metrics import dtw
# from dtaidistance import dtw
# from dtaidistance import dtw_visualisation as dtwvis


os.chdir(r'C:\Users\akshaya.p.kadidal\Desktop\Anomaly Comparison\Anomaly KPIs')

#d1 = pd.read_excel('Core_mavenir_rcs_voice_KPI_CORE_MAVENIR_RCS_VOICE_PERHOUR_GeographyL0_PAN JAPAN_08082020_025901.xlsx', parse_dates=[['Date', 'Time']])
#d1.Date_Time
#d1.index
d1 = pd.DataFrame()

filenames = glob('*.xlsx')

#len(filenames)
#data = pd.read_excel(filenames[13], parse_dates=[['Date', 'Time']])

for f in filenames:
    data = pd.read_excel(f, parse_dates=[['Date', 'Time']])
    data = data.set_index('Date_Time')
    data = data.sort_values(by = 'Date_Time')
    data = data.drop(['Geography','Node'], axis=1, errors='ignore')
    d1 = pd.concat([d1, data], axis=1, join='outer')



# object_columns = d1.select_dtypes(include='object').describe()
# remove_columns = object_columns.loc['unique',:]==1

# columns_to_remove = remove_columns[remove_columns==1].index
# columns_to_remove = columns_to_remove.to_list()
# columns_to_retain = remove_columns[remove_columns!=1].index
# columns_to_retain = columns_to_retain.to_list()
# d1 = d1.drop(columns_to_remove, axis=1)

# for j in columns_to_retain:
#     d1[j] = pd.to_numeric(d1[j],errors='coerce')
    

# d1.info()

# columns = d1.select_dtypes(include=['float64','int64']).columns

# for i in columns:
#     q1 = d1[i].quantile(.25)
#     q3 = d1[i].quantile(.75)
#     iqr = (q3-q1)*1.5
#     ptile99 = d1[i].quantile(.99)
#     ptile01 = d1[i].quantile(.01)
#     skew = d1[i].skew()
#     kurt = d1[i].kurtosis()
#     if (q3 + iqr) > ptile99 :
#         d1.loc[d1[i] > (q3 + iqr),i] = ptile99
#     else :
#         d1.loc[d1[i] > (q3 + iqr),i] = (q3 + iqr)
    
#     if q1 - iqr < ptile01 :
#         d1.loc[d1[i] < (q1 - iqr),i] = ptile01
#     else :
#         d1.loc[d1[i] < (q1 - iqr),i] = (q1 - iqr)


def Clean_ts(df):
    object_columns = df.select_dtypes(include='object').describe()
    remove_columns = object_columns.loc['unique',:]==1
    
    columns_to_remove = remove_columns[remove_columns==1].index
    columns_to_remove = columns_to_remove.to_list()
    columns_to_retain = remove_columns[remove_columns!=1].index
    columns_to_retain = columns_to_retain.to_list()
    df = df.drop(columns_to_remove, axis=1)
    
    for j in columns_to_retain:
        df[j] = pd.to_numeric(df[j],errors='coerce')
        
    
    columns = df.select_dtypes(include=['float64','int64']).columns
    
    #Outlier Treatment Replace any value more than 1.5 of IQR with 99th Percentile value
    for i in columns:
        q1 = df[i].quantile(.25)
        q3 = df[i].quantile(.75)
        iqr = (q3-q1)*1.5
        ptile99 = df[i].quantile(.99)
        ptile01 = df[i].quantile(.01)
        skew = df[i].skew()
        kurt = df[i].kurtosis()
        if (q3 + iqr) > ptile99 :
            df.loc[df[i] > (q3 + iqr),i] = ptile99
        else :
            df.loc[df[i] > (q3 + iqr),i] = (q3 + iqr)
        
        if q1 - iqr < ptile01 :
            df.loc[df[i] < (q1 - iqr),i] = ptile01
        else :
            df.loc[df[i] < (q1 - iqr),i] = (q1 - iqr)
        
        df = df.replace('-',np.NAN)
        #standardize values
        X = StandardScaler().fit_transform(df[i].values.reshape(-1,1))
        df[i] = X
       
    return(df)

dt = Clean_ts(d1)

#get unix epoch time
dt['UnixTime'] = dt.index.astype(np.int64) // 10**9

dt = dt.fillna(0)


evalu = []

for k in range(10):
    
    km = TimeSeriesKMeans(n_clusters=k+2, verbose=True, random_state=23, metric="dtw")
    
    Y = km.fit_predict(dt.T)
    
    evalu.append(silhouette_score(dt.T, Y, metric="dtw"))

# 6 clusteres is best

km = TimeSeriesKMeans(n_clusters=7, verbose=True, random_state=23, metric="dtw")
    
Y = km.fit_predict(dt.T)

c1 =np.where(Y==0)[0].tolist()
c2 =np.where(Y==1)[0].tolist()
c3 =np.where(Y==2)[0].tolist()
c4 =np.where(Y==3)[0].tolist()
c5 =np.where(Y==4)[0].tolist()
c6 =np.where(Y==5)[0].tolist()
c7 =np.where(Y==6)[0].tolist()

dt.iloc[:,c1].plot(subplots=True,legend=False)
dt.iloc[:,c1].columns
dt.iloc[:,c2].plot(subplots=True,legend=False)
dt.iloc[:,c3].plot(subplots=True,legend=False)
dt.iloc[:,c4].plot(subplots=True,legend=False)
dt.iloc[:,c5].plot(subplots=True,legend=False)
dt.iloc[:,c6].plot(subplots=True,legend=False)
c1

years.plot(subplots=True, legend=False)
pyplot.show()

'''
cm = np.zeros((55,55))


for l in range(54):
    for m in range(54):
        cm[l,m] = dtw(dt.iloc[:,l],dt.iloc[:,m])

cm
plt.imshow(cm, cmap='hot', interpolation='nearest')
plt.show()

'''
CM = pd.DataFrame({'KPI1': [],
    'KPI2': [],
    'DTWCorel': [],
    'Corel': []},)

KPI1 = []
KPI2 = []
DTWCorel = []
Corel = []

for l in dt.columns:
    for m in dt.columns:
        KPI1.append(l)
        KPI2.append(m)
        DTWCorel.append(dtw(dt[l],dt[m]))
        Corel.append(dt[l].corr(dt[m]))

CM = pd.DataFrame({'KPI1': KPI1,
    'KPI2': KPI2,
    'DTWCorel': DTWCorel,
    'Corel': Corel})

CM = CM.sort_values(by = 'DTWCorel')

CM.to_csv('tst.csv')
CM =  CM[CM['KPI1']!=CM['KPI2']]

dt[['1006-ERAB Setup Success Rate (%)','5569-Total Call attempt']].plot(subplots=True)

dt[['1028-Retainability','2798-Total Volte Calls']].plot(subplots=True)

#interesting
dt[['1227-VM Disk Utilization','1028-Retainability']].plot(subplots=True)

dt[['1230-VM CPU Average Utilization','1028-Retainability']].plot(subplots=True)

dt[['1018-Overall Handover Success Rate(%)','3038-Test_AvgReqQueue']].plot(subplots=True)

dt[['1042-Memory Average Utilization (%)','1214-VM Memory Utilization']].plot(subplots=True)

dt[['12428-RCS Messaging failure','2815-Total RCS Calls']].plot(subplots=True)

dt[['1360-Memory.Used','1397-Home RRSR - Re-registration success rate of S-CSCF']].plot(subplots=True)


# unrelated but corelated
dt[['8165-# of concurrent calls','1224-traffic_in_min (min)']].plot(subplots=True)
dt[['1508-VM Memory Average Utilization','8161-# of registered IMS subs (#)']].plot(subplots=True)
dt[['1236-VM Memory Average Utilization','5296-A-SBC Register User Number']].plot(subplots=True)
