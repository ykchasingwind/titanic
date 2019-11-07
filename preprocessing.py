#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import numpy as np
import pandas as pd


def age_encoder(age):
    res = []
    for i in age:
        if i <= 18:
            res.append(0)
        elif 18 < i <= 60:
            res.append(1)
        else:
            res.append(2)
    return res


def sibsp_encoder(sibsp):
    res = []
    for i in sibsp:
        if i == 0:
            res.append(0)
        elif i == 1:
            res.append(1)
        elif i == 2:
            res.append(2)
        else:
            res.append(3)
    return res


def parch_encoder(parch):
    res = []
    for i in parch:
        if i == 0:
            res.append(0)
        elif i == 1:
            res.append(1)
        elif i == 2:
            res.append(2)
        else:
            res.append(3)
    return res


def fare_encoder(fare):
    res = []
    for i in fare:
        if i <= 50:
            res.append(0)
        elif 50 < i <= 100:
            res.append(1)
        else:
            res.append(2)
    return res


tr_data = pd.read_csv('E:/kaggle_games/titanic/Data/input/train.csv')
te_data = pd.read_csv('E:/kaggle_games/titanic/Data/input/test.csv')

tr_label = pd.DataFrame(tr_data['Survived'], columns=['Survived'])
tr_feature = tr_data.drop(['Survived', 'Ticket', 'PassengerId'], axis=1)

tr_feature['Age'] = tr_feature['Age'].fillna(tr_feature['Age'].median())
tr_feature['Age'] = age_encoder(tr_feature['Age'])

tr_feature['SibSp'] = sibsp_encoder(tr_feature['SibSp'])

tr_feature['Parch'] = parch_encoder(tr_feature['Parch'])

tr_feature['Embarked'] = tr_feature['Embarked'].fillna('S')
embarked_mapping = {'S': 0, 'C': 1, 'Q': 2}
tr_feature['Embarked'] = tr_feature['Embarked'].map(embarked_mapping)

tr_feature['Fare'] = fare_encoder(tr_feature['Fare'])

# Cabin中缺省的标为0，否则为1
tr_feature['Cabin'][tr_feature['Cabin'].notnull()] = 1
tr_feature['Cabin'][tr_feature['Cabin'].isnull()] = 0

# 从Name中提取Title特征
tr_feature['Title'] = [x.split(',')[1].split('.')[0] for x in tr_feature['Name']]
tr_feature['Title'][~tr_feature['Title'].str.contains('Mr|Mrs|Miss|Master')] = 'others'

# 进行独热编码，增加特征
tr_feature = tr_feature.join(pd.get_dummies(tr_feature['Title'], prefix='Title'))
tr_feature = tr_feature.join(pd.get_dummies(tr_feature['Sex'], prefix='Sex'))
tr_feature = tr_feature.join(pd.get_dummies(tr_feature['Embarked'], prefix='Embarked'))
tr_feature = tr_feature.join(pd.get_dummies(tr_feature['Pclass'], prefix='Pclass'))

# 根据SibSp和Parch提取是否与家人一起的Isalone特征
tr_feature['Isalone'] = np.nan
tr_feature['Isalone'][tr_feature['SibSp'] + tr_feature['Parch'] == 0] = 1
tr_feature['Isalone'][tr_feature['Isalone'].isnull()] = 0

tr_feature = tr_feature.drop(['Name', 'Sex', 'Embarked', 'Title'], axis=1)

tr_label.to_csv('E:/kaggle_games/titanic/tr_label.csv')
tr_feature.to_csv('E:/kaggle_games/titanic/tr_feature.csv')






