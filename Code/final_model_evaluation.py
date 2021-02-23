import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import pickle 


train_set=pd.read_csv('final_train_set.csv')
test_set=pd.read_csv('final_test_set.csv')

X_train=train_set.drop('plan',axis=1)
y_train=train_set['plan']

X_test=test_set.drop('plan',axis=1)
y_test=test_set['plan']


mymodel=pickle.load(open('model.pkl', 'rb'))


###### XGB ######

mymodel.fit(X_train,y_train)
y_best_xgb=mymodel.predict(X_test)



print(classification_report(y_test,y_best_xgb))




X_test_predictions=pd.read_csv('final_X_test.csv')

mymodel.fit(X_train,y_train)
y_predictions=mymodel.predict(X_test_predictions)

y_predictions=pd.Series(y_predictions).rename('Predictions')
y_predictions.value_counts()



users_final=pd.read_csv('users_test.csv')
users_id=users_final['user_id']


final_predictions=pd.concat([users_id,y_predictions],axis=1)


final_predictions.to_csv('final_predictions.csv', index=False)








