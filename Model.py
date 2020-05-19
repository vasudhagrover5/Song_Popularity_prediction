# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 11:25:37 2019

@author: Ritwik Gupta
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,precision_score
from sklearn.metrics import roc_auc_score,f1_score,recall_score,confusion_matrix

import os
os.chdir('/media/ritwik/New Volume/Study Material/Machine Learning/Project/Datasets')


dataset = pd.read_csv('Million_final.csv').drop('Unnamed: 0',axis=1)
features1 = dataset.drop(['analysis_sample_rate','artist_7digitalid','artist_latitude','artist_longitude','song_id','track_7digitalid','track_id','transfer_note','artist_id','artist_mbid','artist_playmeid','artist_mbtags','artist_mbtags_count','audio_md5','release_7digitalid','similar_artists','title','song_hotttnesss','artist_terms','artist_terms_freq','artist_terms_weight','segments_timbre','release','artist_location','artist_name'],axis=1)
features = pd.read_csv('Feature.csv').drop('Unnamed: 0',axis=1)
features.columns = features1.columns
labels = dataset['song_hotttnesss']
scores = []

#Splitting labels and features
features_train,features_test,labels_train,labels_test = train_test_split(features,labels,test_size = 0.15)

#Handling Data imbalance using Smote
from imblearn.over_sampling import SMOTE
sm = SMOTE(random_state=2)
features_train, labels_train = sm.fit_sample(features_train, labels_train.ravel())


################################################# Score Calculation #################################################

def calculate(labels_pred):
    accuracy = accuracy_score(labels_test,labels_pred)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))
    scores.append(accuracy * 100.0)    
    
################################################# #XGB approach #######################################################################

from xgboost import XGBClassifier
xgb_clf = XGBClassifier(booster = 'gbtree',eta=2)
xgb_clf.fit(features_train,labels_train)
labels_pred = xgb_clf.predict(features_test)
calculate(labels_pred)

################################################# #LDA approach #######################################################################

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
lda_clf = LinearDiscriminantAnalysis()
lda_clf.fit_transform(features, labels)
labels_pred = lda_clf.predict(features_test)
calculate(labels_pred)
f1_score(labels_test, labels_pred,average='micro')

################################################# DecisionTree approach #######################################################################

from sklearn.tree import DecisionTreeClassifier  
dt_clf = DecisionTreeClassifier()  
dt_clf.fit(features_train, labels_train)
labels_pred = dt_clf.predict(features_test)
calculate(labels_pred)
f1_score(labels_test, labels_pred, average='weighted')

################################################# RandomForest approach #######################################################################

from sklearn.ensemble import RandomForestClassifier
rf_clf = RandomForestClassifier(n_estimators=50, random_state=0)  
rf_clf.fit(features_train, labels_train)  
labels_pred = rf_clf.predict(features_test)
calculate(labels_pred)
f1_score(labels_test, labels_pred, average='weighted')

################################################# SVM approach #######################################################################

from sklearn.svm import SVC
svc_clf = SVC(kernel = 'linear', random_state = 7)
svc_clf.fit(features_train, labels_train)
labels_pred = svc_clf.predict(features_test)
f1_score(labels_test, labels_pred, average='weighted')
calculate(labels_pred)
################################################# LogisticRegression #######################################################################

from sklearn.linear_model import LogisticRegression
lr_clf = LogisticRegression()
lr_clf.fit(features_train, labels_train)
labels_pred = lr_clf.predict(features_test)
f1_score(labels_test, labels_pred, average='weighted')
calculate(labels_pred)

################################################# Adaboost #######################################################################

from sklearn.ensemble import AdaBoostClassifier
ab_clf = AdaBoostClassifier(n_estimators = 56,random_state=7,base_estimator=rf_clf,algorithm='SAMME')
ab_clf.fit(features_train,labels_train)
labels_pred = ab_clf.predict(features_test)
f1_score(labels_test, labels_pred, average='weighted')
calculate(labels_pred)

################################################# Ensemble #######################################################################


from sklearn.ensemble import VotingClassifier
clf_vote = VotingClassifier(estimators = [('XGBoost',xgb_clf),('LDA',lda_clf),('Decision Tree',dt_clf),('Random Forest',rf_clf),('SVC',svc_clf),('Logistic regression',lr_clf),('AdaBoost',ab_clf)])
clf_vote.fit(features_train,labels_train)
labels_train = labels_train.reshape(15944)
pred_vote = clf_vote.predict(features_test)
f1_score(labels_test,pred_vote,average='micro')
calculate(labels_pred)

#Plotting the scores
approach = ['XGB','LDA','DecisionTree','RandomForest','SVC','Logistic Regression','Adaboost','Ensemble']
import matplotlib.pyplot as plt
plt.bar(approach,scores)
plt.xticks(rotation=35)

#Classification Report
from sklearn.metrics import classification_report
report = classification_report(labels_test,pred_vote)
print(report)


from sklearn.model_selection import cross_val_score
score = cross_val_score(clf_vote,features_test,labels_test, cv=10)
print("Accuracy: %0.2f (+/- %0.2f)" % (score.mean(), score.std() * 2))
f1_score(labels_test, labels_pred, average='weighted')

#Creating Pickle file
import pickle
with open('model','wb') as f:
    pickle.dump(xgb_clf,f)    

