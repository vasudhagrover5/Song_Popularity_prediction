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
features = dataset.drop(columns = ['analysis_sample_rate','artist_7digitalid','artist_latitude','artist_longitude','song_id','track_7digitalid','track_id','transfer_note','artist_id','artist_mbid','artist_playmeid','artist_mbtags','artist_mbtags_count','audio_md5','release_7digitalid','similar_artists','title','song_hotttnesss','artist_terms','artist_terms_freq','artist_terms_weight','segments_timbre','release','artist_location','artist_name'])
features.info()
labels = dataset['song_hotttnesss']

#from sklearn.preprocessing import StandardScaler
#sc = StandardScaler()  
#features = sc.fit_transform(features)
#Label Encoding of the temprory features
#from sklearn.preprocessing import LabelEncoder
#le = LabelEncoder()
#features['artist_id'] = features['artist_id'].astype(str)
#features['artist_id'] = le.fit_transform(features['artist_id'])
"""
mode = []
for i, row in features.iterrows():
    if features['mode'][i] == 1:
        mode.append(features['mode_confidence'][i])
    else:
        mode.append(-features['mode_confidence'][i])
    print(mode[i])
features = features.drop(['mode_confidence','mode'],axis=1)
features['mode'] = mode
"""
from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt
model = ExtraTreesClassifier()
model.fit(features,labels)
print(model.feature_importances_) #use inbuilt class feature_importances of tree based classifiers
#plot graph of feature importances for better visualization
feat_importances = pd.Series(model.feature_importances_, index=features.columns)
feat_importances.nlargest(30).plot(kind='barh')
plt.show()

features = features.drop(columns = ['danceability','energy'],axis=1)
features.to_csv('Features_List.csv')
"""
mode = []
for i, row in features.iterrows():
    if features['mode'][i] == 1:
        mode.append(features['mode_confidence'][i])
    else:
        mode.append(-features['mode_confidence'][i])
    print(mode[i])
features = features.drop(['mode_confidence','mode'],axis=1)
features['mode'] = mode"""        