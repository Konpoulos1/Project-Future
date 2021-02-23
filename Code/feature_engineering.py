import pandas as pd
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from imblearn.over_sampling import ADASYN,SMOTE, SMOTENC
from imblearn.under_sampling import TomekLinks,OneSidedSelection

#import csv
the_users=pd.read_csv('the_users.csv')

the_users.drop(['SILVER_ENGAGEMENT_FEES_SAVED','SILVER_ENGAGEMENT_INACTIVE_CARD','CASHBACK','FEE'],axis=1,inplace=True)

# FEE, CASHBACK, SILVER_, SILVER_

the_users_test=pd.read_csv('the_users_test.csv')


the_users.drop(['TRUE','FALSE','UNKNOWN'],axis=1,inplace=True)
the_users_test.drop(['TRUE','FALSE','UNKNOWN'],axis=1,inplace=True)


devices=pd.read_csv('devices.csv')
devices_test=pd.read_csv('devices_test.csv')

devices['brand']=np.where(devices['brand']=='Apple',1,0)
devices_test['brand']=np.where(devices_test['brand']=='Apple',1,0)


the_users=the_users.merge(devices, how='left', on='user_id')
the_users_test=the_users_test.merge(devices_test, how='left', on='user_id')





extras=[feature for feature in the_users.columns if feature not in the_users_test.columns]



for feature in extras:
    the_users_test[feature]=0

the_users_test.drop('COMPLETED',axis=1,inplace=True)



the_users_test =the_users_test.reindex(columns=list(the_users.columns.values))
the_users_test.drop(['plan','user_id'],axis=1,inplace=True)

the_users_test.fillna(0,inplace=True)


#the_users.drop

#change the plan to Standard = 0, Gold & Silver = 1
the_users['plan']=np.where(the_users['plan']=='STANDARD',0,1)



'''
   
for feature in features_transformation:
    the_users[feature]=np.sqrt(the_users[feature])
 '''


X=the_users.drop(['plan'],axis=1)
y=the_users['plan']

#Split to create train and test sets
X_train, X_test, y_train ,y_test = train_test_split(X,y, random_state=0, test_size=0.20, stratify=y)


train_set=pd.concat([X_train,y_train],axis=1)
test_set=pd.concat([X_test,y_test],axis=1)



users_id=test_set['user_id'].to_csv('users_id.csv',index=False)

train_set.drop('user_id',axis=1,inplace=True)
test_set.drop('user_id',axis=1,inplace=True)
#country_label
country_labels=train_set.groupby(['country'])['plan'].count().sort_values().index
country_labels={k:i for i,k in enumerate(country_labels,0)}
train_set['country']=train_set['country'].map(country_labels)

test_set['country']=test_set['country'].map(country_labels)
the_users_test['country']=the_users_test['country'].map(country_labels)


#email_label
email_labels=train_set.groupby(['attributes_notifications_marketing_email'])['plan'].count().sort_values().index
email_labels={k:i for i,k in enumerate(email_labels,0)}
train_set['attributes_notifications_marketing_email']=train_set['attributes_notifications_marketing_email'].map(email_labels)

test_set['attributes_notifications_marketing_email']=test_set['attributes_notifications_marketing_email'].map(email_labels)

#notifications_label
notifications_label=train_set.groupby(['attributes_notifications_marketing_push'])['plan'].count().sort_values().index
notifications_label={k:i for i,k in enumerate(notifications_label,0)}
train_set['attributes_notifications_marketing_push']=train_set['attributes_notifications_marketing_push'].map(notifications_label)

test_set['attributes_notifications_marketing_push']=test_set['attributes_notifications_marketing_push'].map(notifications_label)

'''
#label encoding initialization
label_country=LabelEncoder()
label_email=LabelEncoder()
label_push=LabelEncoder()

#label encoding for country and transform test set
train_set['country']=label_country.fit_transform(train_set['country'])
test_set['country']=label_country.transform(test_set['country'])

#label encoding for email and transform test set
train_set['attributes_notifications_marketing_email']=label_email.fit_transform(train_set['attributes_notifications_marketing_email'])
test_set['attributes_notifications_marketing_email']=label_email.transform(test_set['attributes_notifications_marketing_email'])

#label encoding for push and transform test set
train_set['attributes_notifications_marketing_push']=label_push.fit_transform(train_set['attributes_notifications_marketing_push'])
test_set['attributes_notifications_marketing_push']=label_push.transform(test_set['attributes_notifications_marketing_push'])

#one hot encoding for country
ohe_country=OneHotEncoder(handle_unknown='ignore')
country_sparse=ohe_country.fit_transform(train_set.iloc[:,0].values.reshape(-1,1)).toarray()
country_sparse=pd.DataFrame(country_sparse,columns=['Central','Non-Europe','North','South','West'],index=train_set.index)

#one hot encoding for email 
ohe_email=OneHotEncoder(handle_unknown='ignore')
email_sparse=ohe_email.fit_transform(train_set.iloc[:,2].values.reshape(-1,1)).toarray()
email_sparse=pd.DataFrame(email_sparse,columns=['Email_no','Email_yes','missing'],index=train_set.index)

#one hot encoding for push 
ohe_push=OneHotEncoder(handle_unknown='ignore')
push_sparse=ohe_push.fit_transform(train_set.iloc[:,3].values.reshape(-1,1)).toarray()
push_sparse=pd.DataFrame(push_sparse,columns=['Push_yes','Push_no','missing'],index=train_set.index)



#one hot encoding for country in test set
country_sparse_test=ohe_country.transform(test_set.iloc[:,0].values.reshape(-1,1)).toarray()
country_sparse_test=pd.DataFrame(country_sparse_test,columns=['Central','Non-Europe','North','South','West'],index=test_set.index)

#one hot encoding for email in test set
email_sparse_test=ohe_email.transform(test_set.iloc[:,2].values.reshape(-1,1)).toarray()
email_sparse_test=pd.DataFrame(email_sparse_test,columns=['Email_no','Email_yes','missing'],index=test_set.index)

#one hot encoding for push in test set
push_sparse_test=ohe_push.transform(test_set.iloc[:,3].values.reshape(-1,1)).toarray()
push_sparse_test=pd.DataFrame(push_sparse_test,columns=['Push_yes','Push_no','missing'],index=test_set.index)


#remove columns
train_set.drop(['attributes_notifications_marketing_email','country','attributes_notifications_marketing_push'],axis=1, inplace=True)
test_set.drop(['attributes_notifications_marketing_email','country','attributes_notifications_marketing_push'],axis=1, inplace=True)

#train and test set after encoding
train_set=pd.concat([train_set,country_sparse,email_sparse,push_sparse],axis=1)
test_set=pd.concat([test_set,country_sparse_test,email_sparse_test,push_sparse_test],axis=1)
'''







#scaling 
scaler=StandardScaler()
scaled_X_train=scaler.fit_transform(train_set.drop('plan',axis=1))

scaled_X_test=scaler.transform(test_set.drop('plan',axis=1))

scaled_test_users=scaler.transform(the_users_test)

scaled_test_users=pd.DataFrame(scaled_test_users, columns=the_users_test.columns)

#UNDERSAMPLING and/or OVERSAMPLING

#undersampling the train set
under=OneSidedSelection()
X_train_res, y_train_res=under.fit_resample(scaled_X_train, y_train)


#oversampling the train set
sm=SMOTE()
X_train_res, y_train_res= sm.fit_resample(X_train_res, y_train_res)

X_train_res=pd.DataFrame(X_train_res, columns=train_set.drop('plan',axis=1).columns)



#creating the final train and test set for modeling 
train_set=pd.concat([X_train_res, y_train_res],axis=1)

scaled_X_test=pd.DataFrame(scaled_X_test,  columns=test_set.drop('plan',axis=1).columns)

test_set= pd.concat([scaled_X_test, test_set['plan'].reset_index(drop=True)],axis=1)



train_set.to_csv('train_set.csv', index=False)
test_set.to_csv('test_set.csv', index=False)







scaled_test_users.to_csv('scaled_test_users.csv', index=False)






















