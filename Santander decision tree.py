import pandas as pd
import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn import cross_validation
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
import time
from sklearn.cross_validation import KFold
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor


training = pd.read_csv('train.csv', delimiter=',')
test = pd.read_csv('test.csv', delimiter=',')


#remove numerical columns that have constant values
remove = []

for column in training.columns:
    if training[column].std()==0:
        remove.append(column)

#now we will take the training and test data and get rid of these columns
training.drop(remove, axis=1, inplace=True)
test.drop(remove, axis=1, inplace=True)

#removes duplicate columns
colstoremove = []
columns = training.columns
for i in range(len(columns)-1):
    v= training[columns[i]].values
    for j in range(i+1, len(columns)):
        if np.array_equal(v, training[columns[j]].values):
            colstoremove.append(columns[j])
training.drop(colstoremove, axis=1, inplace = True)
test.drop(colstoremove, axis=1, inplace = True)

#split in predictor and target variables
predictors  = training.iloc[:,:-1]
target = training.TARGET

predictors2 = ['var38','var15','saldo_var30','saldo_medio_var5_hace3','saldo_medio_var5_ult3','saldo_medio_var5_hace2','num_var45_ult3','num_var45_hace3','saldo_medio_var5_ult1','num_var22_ult3','saldo_var42','saldo_var5','imp_op_var41_efect_ult3','num_var45_hace2']

#splitting the training set into a train and test dataset by random permutation
np.random.seed(1)
training = training.reindex(np.random.permutation(training.index))
max_row = math.floor(training.shape[0] * 0.8)
train = training.iloc[:max_row]
train_test = training.iloc[max_row:]

pred = train.columns[:-1]

clf = DecisionTreeClassifier(min_samples_split=75, max_depth =4, random_state=1)
clf.fit(train[predictors2], train['TARGET'])
predictions = clf.predict(train_test[predictors2])

error_test = roc_auc_score(predictions, train_test['TARGET'])

print("The roc_auc score is", error_test)


'''
submission = pd.DataFrame({
    'ID': test['ID'],
    'TARGET' : predictions
    })

submission.to_csv('kaggle_submission7.csv', index = False)
'''
