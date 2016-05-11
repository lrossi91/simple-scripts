from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
import pandas as pd

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
testPassId = pd.read_csv('test.csv')


y_train = train['Survived']
x_train= train.drop('Survived', axis=1)

#replace missing values in Age column with the column.mean
x_train['Age'].fillna(x_train['Age'].mean(), inplace=True)
test['Age'].fillna(test['Age'].mean(), inplace=True)
test['Fare'].fillna(test['Fare'].mean(), inplace=True)
#print(x_train.describe())

#select numeric columns by taking only the ones that are not objects datatypes
numeric_variables = list(x_train.dtypes[x_train.dtypes !='object'].index)
#print(x_train[numeric_variables].head())

#drop the columns name, ticket and PassengerId
x_train.drop(['Name','Ticket','PassengerId'],axis=1, inplace=True)
test.drop(['Name','Ticket','PassengerId'], axis=1, inplace=True)

#for cabin we only want to include the first letter of each observation
#for analysis purposes
def clean_cabin(x):
    try:
        return x[0]
    except TypeError:
        return 'None'

x_train['Cabin'] = x_train['Cabin'].apply(clean_cabin)
test['Cabin'] = test['Cabin'].apply(clean_cabin)

categorical_variables = ['Sex','Cabin','Embarked']
for variable in categorical_variables:
    #fill missing data with the word Missing. This will create a new category
    x_train[variable].fillna('Missing', inplace=True)
    test[variable].fillna('Missing', inplace=True)
    #creating arrays of dummy variables
    dummies = pd.get_dummies(x_train[variable], prefix=variable)
    dummies_test = pd.get_dummies(test[variable], prefix=variable)
    #update x_train to include the dummies and drop original variable
    x_train = pd.concat([x_train, dummies], axis=1)
    x_train.drop([variable], axis=1, inplace=True)
    test = pd.concat([test, dummies_test], axis=1)
    test.drop([variable], axis=1, inplace=True)



#lets create the random forest. oob_score is a validation technique
#model = RandomForestRegressor(n_estimators=1000, oob_score = True, random_state=42)
model = RandomForestClassifier(n_estimators=1000,max_depth = 5, random_state=42)

#we will start by only including numerical variables because we have not
#create dummy variables for the categorical ones yet
model.fit(x_train, y_train)
#print(model.oob_score_)

#here is how to create the c-stat with oob. oob gives the probability of each observation
#to either survive or die
#y_oob = model.oob_prediction_
#print('c-stat: ',roc_auc_score(y_train, y_oob))

#two columns are missing from the test data, which are Embarked Missing
#and Cabin_T, we need to add them in order to apply the random forest
test['Embarked_Missing'] = 0
test['Cabin_T'] = 0

#create the predictions for the test data
predictions = model.predict(test)


#create new dataframe for submission
submission = pd.DataFrame({
    'PassengerId': testPassId['PassengerId'],
    'Survived': predictions
    })
submission.to_csv('SubmissionInternetScript.csv', index=False)













