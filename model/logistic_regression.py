#!/usr/bin/env python 
# -*- coding:utf-8 -*-

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

lf2 = LogisticRegression()
clf2.fit(X_train, y_train)
y_predict2 = clf2.predict(X_test)
print(accuracy_score(y_test, y_predict2))
print(confusion_matrix(y_test, y_predict2))
print(classification_report(y_test, y_predict2))



