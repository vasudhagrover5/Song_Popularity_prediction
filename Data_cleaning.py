# -*- coding: utf-8 -*-


#!pip install impyute
from impyute.imputation.cs import mice

def b2s(value):
    value = re.sub(r"^b'",'',value)
    value = re.sub(r'^b','',value)
    value = re.sub(r"'",'',value)
    value = re.sub(r'\"','',value)
    value = re.sub(r" ?\([^)]+\)", "", value)
    value = re.sub(r'\,','',value)
    value = re.sub(r'\’','',value)
    return value

#Importing required libraries
import pandas as pd
import numpy as np
import re

dataset = pd.read_csv('Datasets/Million.csv').drop(['Unnamed: 0'],axis = 1)
df_bb = pd.read_csv('Datasets/Billboardtop100.csv').drop('Unnamed: 0',axis = 1)
df_y = pd.read_csv('Datasets/Years.csv').drop('Unnamed: 0',axis = 1)
df_y['Title'] = df_y['Title'].apply(lambda x: re.sub('[^0-9a-zA-Z\s]','',str(x)))

#Stripping [] and b''
for col in dataset:
    if dataset[col].dtype==object:
        dataset[col]=dataset[col].str.replace("b'","")
        dataset[col]=dataset[col].str.replace("'","")
        dataset[col]=dataset[col].str.strip('[]')

#Cleaning the title column of billboard dataset
df_bb['Title'] = df_bb['Title'].apply(lambda x:b2s(x))
df_bb['Title']= df_bb['Title'].drop_duplicates(keep='first')
df_bb = df_bb.dropna()

#Function to add song popularity from bill board charts
def hits(data):
    if data['title'] in list(df_bb['Title']):
        return 1
    else:
        return 0
dataset['song_hotttnesss'].describe()

dataset['song_hotttnesss'] = dataset['song_hotttnesss'].fillna(0) 

#Classifying songs as popular which have popularity over 75 percentile
for i in range(0,len(dataset['song_hotttnesss'])):
    if dataset['song_hotttnesss'][i]< 0.538:
        dataset['song_hotttnesss'][i] = 0
    elif (dataset['song_hotttnesss'][i] >= 0.538):
        dataset['song_hotttnesss'][i] = 1


df_comp = pd.DataFrame(dataset[dataset['song_hotttnesss']==0]['title'])
df_comp['song_hottness'] = df_comp.apply(hits,axis = 1)
df_comp = df_comp[df_comp['song_hottness']==1]

print(df_comp.shape)

for i in range(0,10000):
    if dataset['title'][i] in list(df_comp['title']):
        dataset['song_hotttnesss'][i] = 1
print(dataset['song_hotttnesss'].value_counts())


def clean_symbols(value):
    for i in range(len(dataset)):
      dataset[value][i] = re.sub(r'  ',' ',str(dataset[value][i]))
      dataset[value][i] = re.sub(r'\[',' ',str(dataset[value][i]))
      dataset[value][i] = re.sub(r'\]',' ',str(dataset[value][i]))
      dataset[value][i] = re.sub(r'\.\.\.',' ',str(dataset[value][i]))
      l1 = str(dataset[value][i]).split()
      mean = np.mean([float(i) for i in l1])
      dataset[value][i] = mean
    
col = ['bars_start','beats_confidence','beats_start','beats_confidence','bars_confidence','sections_confidence','sections_start','segments_confidence','segments_loudness_max','segments_loudness_max_time','segments_loudness_start','segments_start','segments_pitches','tatums_confidence','tatums_start','segments_timbre']
for item in col:
    clean_symbols(item)
    
      
dataset.head()

features = dataset.drop(['analysis_sample_rate','artist_7digitalid','artist_latitude','artist_longitude','song_id','track_7digitalid','track_id','transfer_note','artist_id','artist_mbid','artist_playmeid','artist_mbtags','artist_mbtags_count','audio_md5','release_7digitalid','similar_artists','title','song_hotttnesss','artist_terms','artist_terms_freq','artist_terms_weight','segments_timbre','release','artist_location','artist_name'],axis=1)
features = features.replace('',np.nan)

for i in features.columns:
    if features[i].dtype == 'O':
        features[i] = features[i].astype(float)

# Replacing 0 with nan values as there is a very little chance for mean to be 0. This implies that the values are missing.
for i in col:
    features[i] = features[i].replace(0,np.nan)

features['year'] = features['year'].astype(int)
features['year'] = features['year'].replace(0,np.nan)

#Filling missing values using mice
features_array = mice(np.array(features))

features = pd.DataFrame(features_array,columns=features.columns)
features.year = features.year.astype(int)

features.to_csv('Datasets/Feature_List.csv')
dataset.to_csv('Datasets/Million_final.csv')
