import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score 

titanic_csv = pd.read_csv("Data/titanic.csv")
titanic_csv.head()
titanic_csv.shape

titanic_csv.info()

titanic_csv = titanic_csv[['Survived', 'Age', 'Sex', 'Pclass']]
titanic_csv = pd.get_dummies(titanic_csv, columns=['Sex', 'Pclass'])
titanic_csv.dropna(inplace=True)
titanic_csv.head()

x = titanic_csv.drop('Survived', axis=1)
y = titanic_csv['Survived']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=0)

model = LogisticRegression(random_state=0)
model.fit(x_train, y_train)
model.score(x_test, y_test)

#Крос перекрестная оценка
cross_val_score(model, x, y, cv=5).mean()
