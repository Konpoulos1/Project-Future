import pandas as pd
import numpy as np
from datetime import date

#expanding columns view
pd.set_option('display.max_columns', 500)
#expanding rows view
pd.set_option('display.max_rows', 500)

#importing csv, excel files 
users=pd.read_excel('my_users.xlsx')
transactions_1=pd.read_csv('transactions_1.csv')
transactions_2=pd.read_csv('transactions_2.csv')
transactions_3=pd.read_csv('transactions_3.csv')
devices=pd.read_csv('devices.csv')
notifications=pd.read_csv('notifications.csv')

users.drop('num_contacts',axis=1,inplace=True)

#converting 'created_date' column to datetime type
users['created_date']=pd.to_datetime(users['created_date'])
transactions_1['created_date']=pd.to_datetime(transactions_1['created_date'])
transactions_2['created_date']=pd.to_datetime(transactions_2['created_date'])
transactions_3['created_date']=pd.to_datetime(transactions_3['created_date'])

#concatenate 3 transactions DataFrames to one big DataFrame
transactions=pd.concat([transactions_1,transactions_2,transactions_3])


#function to get age of every user
def get_age(year):
    return 2020-year

#creating new column 'age' for every user
users['age']=np.vectorize(get_age)(users['birth_year'])


#function to find how many days have passed since each transaction
def days_between(mydate):
    return pd.to_datetime(date(2020,6,11))-mydate

#applying days_between to transactions dataframe and creatin a new column 'days_from_trans'
transactions['days_from_trans']=np.vectorize(days_between)(transactions['created_date'])

#applying days_between to users dataframe and creating a new column 'days_of acc'
users['days_of_acc']=np.vectorize(days_between)(users['created_date'])

#remove year column in users dataframe
users.drop(['created_date'],axis=1,inplace=True)

#splitting the column and keep only days value 
transactions['days_from_trans']=transactions['days_from_trans'].astype(str)
transactions['days_from_trans']=transactions['days_from_trans'].str.split(' ').str[0].astype(int)

#splitting the column and keep only days value 
users['days_of_acc']=users['days_of_acc'].astype(str)
users['days_of_acc']=users['days_of_acc'].str.split(' ').str[0].astype(int)


#finding how many days have passed since first and last transaction for every user.
days_from_last_transactions=transactions[transactions['transactions_state']=='COMPLETED'].groupby('user_id')['days_from_trans'].min()
days_from_first_transactions=transactions[transactions['transactions_state']=='COMPLETED'].groupby('user_id')['days_from_trans'].max()


#merging users dataframe with transactions df
users=users.merge(days_from_first_transactions, how='left', on='user_id')
users=users.merge(days_from_last_transactions, how='left', on='user_id')

#rename the columns
users=users.rename(columns={'days_from_trans_x' : 'days_from_first_trans', 'days_from_trans_y' :  'days_from_last_trans'})


#left join users with transactions
merged_transactions=users.merge(transactions, on='user_id', how='left')

#create columns for transactions_state
types_of_transactions_state=[]
for element in transactions['transactions_state'].unique():
    types_of_transactions_state.append(merged_transactions[merged_transactions['transactions_state']==element].groupby('user_id')['transactions_state'].count())

types_of_transactions_state=pd.DataFrame(types_of_transactions_state, index=['COMPLETED', 'REVERTED', 'DECLINED', 'PENDING', 'FAILED','CANCELLED'])
types_of_transactions_state=types_of_transactions_state.T
types_of_transactions_state['user_id'] = types_of_transactions_state.index

#create columns for direction
types_of_direction=[]
for element in transactions['direction'].unique():
    types_of_direction.append(merged_transactions[merged_transactions['direction']==element].groupby('user_id')['direction'].count())

types_of_direction=pd.DataFrame(types_of_direction, index=['OUTBOUND','INBOUND'])
types_of_direction=types_of_direction.T
types_of_direction['user_id'] = types_of_direction.index


merged_transactions['ea_cardholderpresence'].fillna('UNKNOWN', inplace=True)
transactions['ea_cardholderpresence'].fillna('UNKNOWN', inplace=True)



#create column for cardholderpresence
types_of_presence=[]
for element in transactions['ea_cardholderpresence'].unique():
    types_of_presence.append(merged_transactions[merged_transactions['ea_cardholderpresence']==element].groupby('user_id')['ea_cardholderpresence'].count())

types_of_presence=pd.DataFrame(types_of_presence, index=transactions['ea_cardholderpresence'].unique())
types_of_presence=types_of_presence.T
types_of_presence['user_id'] = types_of_presence.index




#create 10 columns of transactions_types
types_of_transactions=[]
for element in transactions['transactions_type'].unique():
    types_of_transactions.append(merged_transactions[merged_transactions['transactions_type']==element].groupby('user_id')['transactions_type'].count())


types_of_transactions=pd.DataFrame(types_of_transactions, index=transactions['transactions_type'].unique())
types_of_transactions=types_of_transactions.T
types_of_transactions['user_id'] = types_of_transactions.index


#merge transactions_type to users
users=users.merge(types_of_transactions,how='left',on='user_id')

#merge transactions_state to users
users=users.merge(types_of_transactions_state,how='left',on='user_id')

#merge direction to users
users=users.merge(types_of_direction, how='left', on='user_id')

#merge presence to users
users=users.merge(types_of_presence, how='left', on='user_id')



#creating 4 metrics sum, mean, frequency, cashback
sum_amount_per_user=transactions[transactions['transactions_state']=='COMPLETED'].groupby('user_id')['amount_usd'].sum()
median_amount_per_user=transactions[transactions['transactions_state']=='COMPLETED'].groupby('user_id')['amount_usd'].median()
frequency_of_transactions=transactions[transactions['transactions_state']=='COMPLETED'].groupby('user_id')['amount_usd'].count()

#joining 4 metrics to users dataframe
users=users.merge(sum_amount_per_user, how='left', on='user_id')
users=users.rename(columns={'amount_usd':'sum_amount_of_transactions'})

users=users.merge(median_amount_per_user, how='left', on='user_id')
users=users.rename(columns={'amount_usd':'median_amount_of_transactions'})

users=users.merge(frequency_of_transactions, how='left', on='user_id')
users=users.rename(columns={'amount_usd':'count_of_transactions'})


users.drop(['birth_year'],axis=1,inplace=True)
#remove created date in transactions dataframe
transactions.drop(['created_date'],axis=1,inplace=True)

#fill nan values with 0
#users['mean_amount_of_transactions'].fillna(0,inplace=True)

#fill nan values with 0
users['count_of_transactions'].fillna(0,inplace=True)

#fill nan values with 0
users['sum_amount_of_transactions'].fillna(0,inplace=True)



features=['TRANSFER', 'CARD_PAYMENT', 'EXCHANGE', 'ATM', 'CARD_REFUND', 'TOPUP',
       'REFUND', 'FEE', 'CASHBACK', 'TAX', 'sum_amount_of_transactions',
       'median_amount_of_transactions', 'count_of_transactions','COMPLETED', 'REVERTED', 'DECLINED', 'PENDING', 'FAILED', 'CANCELLED',
       'user_id',
       'OUTBOUND','INBOUND','FALSE', 'TRUE']



for feature in features:
    users[feature].fillna(0,inplace=True)
   
types_of_reason=[]
for element in notifications['reason'].unique():
    types_of_reason.append(notifications[notifications['reason']==element].groupby('user_id')['reason'].count())

types_of_reason=pd.DataFrame(types_of_reason, index=notifications['reason'].unique())
types_of_reason=types_of_reason.T
types_of_reason['user_id'] = types_of_reason.index   
   
#merge

users=users.merge(types_of_reason,how='left',on='user_id') 

drops=[feature for feature in ['days_from_first_trans', 'days_from_last_trans']]



columns=[feature for feature in users.columns if feature not in drops]

for feature in columns:
    users[feature].fillna(0,inplace=True)

  
merged_transactions_1=merged_transactions.loc[:799999,:]
merged_transactions_2=merged_transactions.loc[799999:1499998,:]
merged_transactions_3=merged_transactions.loc[1499998:,:]

merged_transactions_1.to_csv('merged_transactions_1.csv', index=False)
merged_transactions_2.to_csv('merged_transactions_2.csv',index=False)
merged_transactions_3.to_csv('merged_transactions_3.csv',index=False)


#mapping countries to 4 regions
mapping={'GB' : 'North', 'PL' : 'North', 'FR': 'Central', 'IE':'North','RO' :'Central', 'ES':'West', 'LT' :'North','PT' :'South',
         'MT': 'South' , 'CH' : 'Central','DE':'Central', 'CZ' : 'Central', 'IT' :	'South', 'GR' : 'South', 'CY' : 'South', 'NL' :'Central',
          'LV' : 'North' , 'HU' : 'Central', 'BE' :	'Central', 'SE' :	'North', 'DK' : 'North', 'BG': 'South', 'NO' : 'North',
          'SI' : 'Central',  'AT' :	'Central', 'SK' : 'Central', 'HR' :	'South','JE' : 'Central','GI' : 'South','FI' :	'North',
          'EE' : 'North' , 'LU' : 'Central', 'GG': 'Central', 'GP' : 'Non-Europe', 'IM' :	'North', 'RE' : 'Non-Europe', 'IS' : 'North',
          'LI': 'Central', 'AU' :'Non-Europe', 'MQ': 'Non-Europe'}

users['country']=users['country'].map(mapping)


users['plan']=users['plan'].replace(['GOLD','SILVER'], ['PAID','PAID'])

users=users.dropna()
users.drop('COMPLETED',axis=1,inplace=True)


merged_transactions.to_csv('transactions.txt', sep=',', encoding='utf-8', quotechar='"', decimal=';' , index=False)

