import pandas as pd
import numpy as np
from datetime import date
import time 

t1=time.time()

#expanding columns view
pd.set_option('display.max_columns', 500)
#expanding rows view
pd.set_option('display.max_rows', 500)

#importing csv, excel files 
users=pd.read_csv('users.csv')
transactions_1=pd.read_csv('transactions_1.csv')
transactions_2=pd.read_csv('transactions_2.csv')
transactions_3=pd.read_csv('transactions_3.csv')
devices=pd.read_csv('devices.csv')
notifications=pd.read_csv('notifications.csv')

users.drop('num_contacts',axis=1,inplace=True)

#users=users.merge(devices, how='left', on='user_id')

#converting 'created_date' column to datetime type
users['created_date']=pd.to_datetime(users['created_date'])
transactions_1['created_date']=pd.to_datetime(transactions_1['created_date'])
transactions_2['created_date']=pd.to_datetime(transactions_2['created_date'])
transactions_3['created_date']=pd.to_datetime(transactions_3['created_date'])

#concatenate 3 transactions DataFrames to one big DataFrame
transactions=pd.concat([transactions_1,transactions_2,transactions_3])



###################### EDA ######################


users.info()

categorical_feature=[feature for feature in users.columns if users[feature].dtype !='O']

numerical_features=[feature for feature in users.columns if users[feature].dtype =='O']


features_with_nan=[feature for feature in users.columns if users[feature].isnull().sum()>0]


for feature in features_with_nan:
    print(feature, round(users[feature].isnull().mean()*100,2),'% missing')
    
for feature in categorical_feature:
    print(feature, users[feature].nunique())
    
    
    
    
