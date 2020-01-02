# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 20:55:53 2019

@author: Ritwik Gupta
"""
import pandas as pd
from bs4 import BeautifulSoup as bs
import requests
import json
import time

dataset = pd.read_csv('Million_song_dataset.csv').drop('Unnamed: 0',axis=1)
new_dataset=dataset[dataset['year']==0.0].reset_index()
new_dataset = new_dataset.drop('index',axis=1)
year = []

for i in range(0,10):
    url = "http://musicbrainz.org/ws/2/release-group?artist="+str(new_dataset['artist_mbid'][i])+"&fmt=json"
    response = requests.get(url)
    json_data = response.json()
    year.append(json_data.get('release-groups'))
    time.sleep(0.2)

d1={}
for i in range(0,len(year)):
    if type(year[i]) == list:
        for j in range(0,len(year[i])):
            d1[year[i][j].get('title')] = (year[i][j].get('first-release-date').split('-')[0])
"""
df3 = pd.DataFrame()
df3['title'] = d1.keys()
df3['year'] = d1.values()
df3 = df3[df3['year'] != '']
df3.to_csv('Years.csv')
"""
df3 = pd.read_csv('Years.csv')
df3 = df3.drop('Unnamed: 0',axis=1).dropna().reset_index()
df3 = df3.drop('index',axis=1)
df3.columns = ['title','year']
dataset = pd.merge(dataset,df3,on='title',how='left')
for i in range(0,10000):
    if dataset['year_x'][i] == 0:
        dataset['year_x'][i] = dataset['year_y'][i]
dataset = dataset.drop('year_y',axis=1)
dataset = dataset.rename(index=str,columns ={"year_x":"year"})
dataset['year'].value_counts()
dataset.to_csv('Million_song_dataset.csv')
"""
#new = set(dataset['title']).intersection(set(df3['Title']))

d2={}
for item in new:
    d2[item] = d1[item]
"""
