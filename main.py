# Importing The Libraries
import pandas as pd
import numpy as np

# Importing The Dataset
dataset = pd.read_csv('data/IRIS.csv')

# Splitting Data into Features and Labels
X = dataset.iloc[:, :-1]
y = dataset.iloc[:, -1]

# Splitting Data into Training set and Testing Set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# Creating The Classifier
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(splitter="best", max_depth=30)
clf.fit(X_train, y_train)

# Predicting The Labels of the Test set
y_pred = clf.predict(X_test)

# Calculating The Accuracy Score
from sklearn.metrics import accuracy_score
accuracy_ = accuracy_score(y_pred, y_test)
print(accuracy_)

# Accuracy Score = 0.9736842105263158
