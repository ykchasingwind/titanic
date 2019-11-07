#!/usr/bin/env python 
# -*- coding:utf-8 -*-

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

clf4 = RandomForestClassifier(max_depth=4, n_estimators=100, random_state=0)
clf4.fit(X_train, y_train)
y_predict4 = clf4.predict(X_test)
print(accuracy_score(y_test, y_predict4))
print(clf4.feature_importances_)
