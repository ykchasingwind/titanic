#!/usr/bin/env python 
# -*- coding:utf-8 -*-

from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score

clf3 = AdaBoostClassifier(n_estimators=50, random_state=0)
clf3.fit(X_train, y_train)
y_predict3 = clf3.predict(X_test)
print(accuracy_score(y_test, y_predict3))
print(clf3.feature_importances_)

