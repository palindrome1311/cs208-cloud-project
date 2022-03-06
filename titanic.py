import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
#import seaborn as sns
#import matplotlib.pyplot as plt
#import warnings
from sklearn.model_selection import train_test_split, cross_val_score

from sklearn.tree import DecisionTreeClassifier


import datetime

train = pd.read_csv('titanic_train.csv')
test = pd.read_csv('titanic_test.csv')


train_len = len(train)
# combine two dataframes
df = pd.concat([train, test], axis=0)
df = df.reset_index(drop=True)
## find the null values
df.isnull().sum()
# drop or delete the column
df = df.drop(columns=['Cabin'], axis=1)

# fill missing values using mean of the numerical column
df['Age'] = df['Age'].fillna(df['Age'].mean())
df['Fare'] = df['Fare'].fillna(df['Fare'].mean())

# fill missing values using mode of the categorical column
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])


df['Fare'] = np.log(df['Fare']+1)

## drop unnecessary columns
df = df.drop(columns=['Name', 'Ticket'], axis=1)

cols = ['Sex', 'Embarked']
le = LabelEncoder()

for col in cols:
    df[col] = le.fit_transform(df[col])


train = df.iloc[:train_len, :]
test = df.iloc[train_len:, :]


# input split
X = train.drop(columns=['PassengerId', 'Survived'], axis=1)
y = train['Survived']

# classify column
def classify(model):
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    model.fit(x_train, y_train)
    #print('Accuracy:', model.score(x_test, y_test))
    
    score = cross_val_score(model, X, y, cv=5)
    #print('CV Score:', np.mean(score))



print('----------TRAINING TITANIC DATASET USING LOGISTIC REGRESSION-------------')
a = datetime.datetime.now()
model = DecisionTreeClassifier()
classify(model)
b = datetime.datetime.now()
c=b-a
print('----------TRAINING ENDED -----------')
print('Time Required in seconds - ',c.total_seconds())