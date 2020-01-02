# -*- coding: utf-8 -*-
"""
Created on Wed Dec 25 12:07:15 2019

@author: Ritwik Gupta
"""

import pandas as pd
import numpy as np
import re
dataset = pd.read_csv('Million_song_dataset.csv').drop('Unnamed: 0',axis=1)

def b2s(value):
    value = re.sub(r"^b'",'',value)
    value = re.sub(r'^b','',value)
    value = re.sub(r"'",'',value)
    value = re.sub(r'\"','',value)
    value = re.sub(r" ?\([^)]+\)", "", value)
    value = re.sub(r'\,','',value)
    value = re.sub(r'\’','',value)
    return value
#Checking for null values in the dataset
l1 = {}
l2 = []
for item in dataset:
    l1[item] = dataset[item].isnull().value_counts()
    if l1[item].size==2:
        l2.append(item)

features = dataset.drop(columns=['analysis_sample_rate','artist_7digitalid','artist_latitude','artist_longitude'])
for col in dataset:
    if dataset[col].dtype==object:
        dataset[col]=dataset[col].str.replace("b'","")
        dataset[col]=dataset[col].str.replace("'","")
        dataset[col]=dataset[col].str.strip('[]')

name = list(dataset['artist_name'].value_counts().index[:-1])
score = dataset['artist_name'].value_counts()
dataset['artist_mbtags'].isnull().value_counts()

dataset = dataset.drop(columns=['song_id','track_7digitalid','track_id','transfer_note'])
dataset.to_csv('subset.csv')

#Artist dataset
artist = pd.read_csv('artists.csv')
artist['tags_mb'].isnull().value_counts()
artist['country_mb'] = artist['country_mb'].dropna()
artist.to_csv('artists.csv')

info = pd.read_csv('info.csv').drop("Unnamed: 0",axis=1)
for i in range(len(info)):
    info['location'][i] = re.sub(r"[0-9]+","",info['location'][i])
    info['location'][i] = re.sub(r"^[0-9]+","",info['location'][i])
    info['tags'][i] = re.sub(r"^[0-9]+","",info['tags'][i])
    info['location'][i] = re.sub(r"Name: country_mb, dtype: object","",info['location'][i])
    info['tags'][i] = re.sub(r"Name: tags_mb, dtype: object","",info['tags'][i])
info.to_csv('info.csv')

loc = {}
info['artist'] = name
    #dataset = pd.DataFrame(artist[artist['artist_mb'] == i])
from numpy import nan
import numpy

for i in range(len(dataset)):
    dataset[['artist_location']][i] = info[info['artist']==dataset['artist_name'][i]]['location']
    dataset[['artist_mbtags']][i] = info[info['artist']==dataset['artist_name'][i]]['tags']


dataset['artist_location'].replace("",np.nan)
dataset['artist_mbtags'].replace("",np.nan)
l1=[]
l2=[]
l1 = dataset['artist_terms_freq'].str.replace(r"1.",'')
l2 = l1.str.strip(" ")
l2 = l2.str.replace("\n","")
dataset['artist_terms_freq'] = l2    

l1 = dataset['artist_terms_weight'].str.replace(r"1.",'')
l2 = l1.str.strip(" ")
l2 = l2.str.replace("\n","")
dataset['artist_terms_weight'] = l2

dataset.to_csv("subset.csv")

dataset['song_hotttnesss'].isnull().value_counts()
dataset['song_hotttnesss'] = dataset['song_hotttnesss'].dropna(axis=0)
dataset = dataset[pd.notnull(dataset['song_hotttnesss'])]

