import pandas as pd
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt
from sklearn.feature_selection import RFECV
from sklearn.model_selection import GridSearchCV


train_set=pd.read_csv('train_set.csv')
test_set=pd.read_csv('test_set.csv')


X=train_set.drop('plan', axis=1)
y=train_set['plan']



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33 , random_state=0)


##################### FEATURE SELECTION WITH EXTRA TREES CLASSIFIER #####################




model = ExtraTreesClassifier(random_state=0)
model.fit(X_train,y_train)

feat_importances = pd.Series(model.feature_importances_, index=X_train.columns)
feat_importances.nlargest(X_train.shape[1]).plot(kind='barh')
plt.show()

a=pd.Series(feat_importances.nlargest(25))


X_features=X.copy()

for feature in X_features.columns:
    if feature not in a.index:
        X_features.drop(feature,axis=1,inplace=True)

y_features=y.copy()

final_train_set=pd.concat([X_features,y_features],axis=1)


test_set_features=test_set.drop('plan',axis=1)
test_set_y=test_set['plan']

for feature in test_set_features.columns:
    if feature not in a.index:
        test_set_features.drop(feature,axis=1,inplace=True)



final_test_set=pd.concat([test_set_features,test_set_y],axis=1)


final_train_set.to_csv('final_train_set.csv', index=False)
final_test_set.to_csv('final_test_set.csv' ,index=False)




the_users_test_reduction=pd.read_csv('scaled_test_users.csv')

for feature in the_users_test_reduction.columns:
    if feature not in a.index:
        the_users_test_reduction.drop(feature,axis=1,inplace=True)




the_users_test_reduction.to_csv('final_X_test.csv',index=False)
