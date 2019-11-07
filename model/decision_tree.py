#!/usr/bin/env python 
# -*- coding:utf-8 -*-

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

clf1 = DecisionTreeClassifier(max_depth=4)
clf1.fit(X_train, y_train)
y_predict1 = clf1.predict(X_test)
print(accuracy_score(y_test, y_predict1))

print(confusion_matrix(y_test, y_predict1))

print(classification_report(y_test, y_predict1))
