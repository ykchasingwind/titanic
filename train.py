#!/usr/bin/env python 
# -*- coding:utf-8 -*-


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

tr_label = pd.read_csv('E:/kaggle_games/titanic/tr_label.csv')
tr_feature = pd.read_csv('E:/kaggle_games/titanic/tr_feature.csv')

X_train, X_test, y_train, y_test = train_test_split(tr_feature.iloc[:, 1:], tr_label.iloc[:, 1], test_size=0.3, random_state=0)

# svm
clf = SVC(kernel='linear')
clf.fit(X_train, y_train)
y_predict = clf.predict(X_test)
print(accuracy_score(y_test, y_predict))
print(confusion_matrix(y_test, y_predict))
print(classification_report(y_test, y_predict))

# decision_tree
clf1 = DecisionTreeClassifier(max_depth=4)
clf1.fit(X_train, y_train)
y_predict1 = clf1.predict(X_test)
print(accuracy_score(y_test, y_predict1))
print(confusion_matrix(y_test, y_predict1))
print(classification_report(y_test, y_predict1))

# logistic_regression
clf2 = LogisticRegression()
clf2.fit(X_train, y_train)
y_predict2 = clf2.predict(X_test)
print(accuracy_score(y_test, y_predict2))
print(confusion_matrix(y_test, y_predict2))
print(classification_report(y_test, y_predict2))

# adaboost
clf3 = AdaBoostClassifier(n_estimators=50, random_state=0)
clf3.fit(X_train, y_train)
y_predict3 = clf3.predict(X_test)
print(accuracy_score(y_test, y_predict3))
print(clf3.feature_importances_)

# random_forest
clf4 = RandomForestClassifier(max_depth=4, n_estimators=100, random_state=0)
clf4.fit(X_train, y_train)
y_predict4 = clf4.predict(X_test)
print(accuracy_score(y_test, y_predict4))
print(clf4.feature_importances_)
