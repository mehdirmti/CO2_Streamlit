import pandas as pd


df=pd.read_csv("train.csv")

df = df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
y = df['Survived']
X_cat = df[['Pclass', 'Sex',  'Embarked']]
X_num = df[['Age', 'Fare', 'SibSp', 'Parch']]

for col in X_cat.columns:
    X_cat[col] = X_cat[col].fillna(X_cat[col].mode()[0])
for col in X_num.columns:
    X_num[col] = X_num[col].fillna(X_num[col].median())

X_cat_scaled = pd.get_dummies(X_cat, columns=X_cat.columns)
X = pd.concat([X_cat_scaled, X_num], axis = 1)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train[X_num.columns] = scaler.fit_transform(X_train[X_num.columns])
X_test[X_num.columns] = scaler.transform(X_test[X_num.columns])

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import joblib

def prediction(classifier):
    if classifier == 'Random Forest':
        clf = RandomForestClassifier()
        model = "RF"
    elif classifier == 'SVC':
        clf = SVC()
        model = "SVC"
    elif classifier == 'Logistic Regression':
        clf = LogisticRegression()
        model = "LR"
    clf.fit(X_train, y_train)
    
    # Save model
    joblib.dump(clf, model)
    return clf

for option in ['Random Forest', 'SVC', 'Logistic Regression']:
    clf = prediction(option)