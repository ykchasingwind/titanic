#!/usr/bin/env python 
# -*- coding:utf-8 -*-

from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score


clf = SVC(kernel='linear')
clf.fit(X_train, y_train)
y_predict = clf.predict(X_test)
print(accuracy_score(y_test, y_predict))
print(confusion_matrix(y_test, y_predict))
print(classification_report(y_test, y_predict))