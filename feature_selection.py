# -*- coding: utf-8 -*-
"""
Created on Sun Jun 23 14:38:22 2019

@author: Ritwik Gupta
"""
#Importing the dataset using Pandas
import pandas as pd
import numpy as np
dataset = pd.read_csv('Million_song_dataset.csv').drop('Unnamed: 0',axis=1)

#Checking for null values in the dataset
l1 = {}
for item in dataset:
    l1[item] = dataset[item].isnull().value_counts()
#Temprory feature variable 

f1  = dataset.drop(['danceability','energy','analysis_sample_rate','artist_7digitalid','artist_latitude','artist_longitude','song_id','track_7digitalid','track_id','transfer_note','artist_id','artist_mbid','artist_playmeid','artist_mbtags','artist_mbtags_count','audio_md5','release_7digitalid','similar_artists','title','song_hotttnesss','artist_terms','artist_terms_freq','artist_terms_weight','segments_timbre','release','artist_location','artist_name'],axis=1)

features = pd.read_csv('Feature_List.csv').drop('Unnamed: 0',axis=1)
features.columns = f1.columns
features.info()
labels = dataset['song_hotttnesss']

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()  
scaled_f = sc.fit_transform(features)  

features = pd.DataFrame(scaled_f,columns=features.columns)

"""
from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt
model = ExtraTreesClassifier()
model.fit(features,labels)
print(model.feature_importances_) #use inbuilt class feature_importances of tree based classifiers
#plot graph of feature importances for better visualization
feat_importances = pd.Series(model.feature_importances_, index=features.columns)
feat_importances.nlargest(20).plot(kind='barh')
plt.show()

"""


from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
rfe_selector = RFE(estimator=LogisticRegression(), n_features_to_select=20, step=10, verbose=5)
rfe_selector.fit(features, labels)
rfe_support = rfe_selector.get_support()
rfe_feature = features.loc[:,rfe_support].columns.tolist()
print(str(len(rfe_feature)), 'selected features')

features = features.loc[:,rfe_feature]

features = features.drop(columns = ['danceability','energy','bars_start'],axis=1)
features.to_csv('Features_List.csv')
