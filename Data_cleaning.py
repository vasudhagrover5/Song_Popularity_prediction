# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 12:11:34 2019

@author: Ritwik Gupta
"""
#Function to apply regx
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

#Importing the dataset into dataframes
dataset = pd.read_csv('Million_song_dataset.csv').drop(['Unnamed: 0.1','Unnamed: 0'],axis = 1)
df_bb = pd.read_csv('Billboard_top100.csv').drop('Unnamed: 0',axis = 1)
df_y = pd.read_csv('Years.csv').drop('Unnamed: 0',axis = 1)
df_y['Title'] = df_y['Title'].apply(lambda x: re.sub('[^0-9a-zA-Z\s]','',str(x)))
df_y.to_csv('Years.csv')
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

#Recreating the hotness column
def hits(data):
    if data['title'] in list(df_bb['Title']):
        return 1
    else:
        return 0
new = pd.read_csv('Million.csv').drop('Unnamed: 0',axis=1)
dataset['song_hotttnesss'] = dataset['song_hotttnesss'].fillna(0) 

for i in range(0,len(dataset['song_hotttnesss'])):
    if dataset['song_hotttnesss'][i]< 0.75:
        dataset['song_hotttnesss'][i] = 0
    elif (dataset['song_hotttnesss'][i] >= 0.75):
        dataset['song_hotttnesss'][i] = 1
             
df_comp = pd.DataFrame(dataset[dataset['song_hotttnesss']==0]['title'])
df_comp['song_hottness'] = df_comp.apply(hits,axis = 1)
df_comp = df_comp[df_comp['song_hottness']==1]

for i in range(0,10000):
    if dataset['title'][i] in list(df_comp['title']):
        dataset['song_hotttnesss'][i] = 1
        
#dataset['song_hotttnesss'].value_counts()
"""
l1 = dataset['artist_terms_freq'].str.replace(r"1.",'')
l2 = l1.str.strip(" ")
l2 = l2.str.replace("\n","")
dataset['artist_terms_freq'] = l2    

l1 = dataset['artist_terms_weight'].str.replace(r"1.",'')
l2 = l1.str.strip(" ")
l2 = l2.str.replace("\n","")
dataset['artist_terms_weight'] = l2
"""
count = []
beats = []
def mean(value,i):
    dataset[value][i] = np.mean(np.array(str(dataset[value][i]).split()).astype(np.float)) 
def zero(value,i):
    dataset[value][i] = str(dataset[value][i]).replace("0.  ","0.0")
def dot(value,i):
    dataset[value][i] = re.sub("...",'',dataset[value][i])
def fill(value):
    dataset[value] = dataset[value].fillna(np.mean(dataset[value]))    


for i in range(len(dataset)):
    try:    
        mean('bars_confidence',i)
    except:
        count.append(i)
    try:
        zero('beats_confidence',i)
        mean('beats_confidence',i)
    except:
        beats.append(i)
for i in(beats):
    dot('beats_confidence',i)
    try:    
        zero('beats_confidence',i)
        mean('beats_confidence',i)
    except:
        beats.append(i)
fill('beats_confidence')


#Check for ... in beats and bar confidence
for i in(count):
    dot('bars_confidence',i)
    try:    
        zero('bars_confidence',i)
        mean('bars_confidence',i)
    except:
        count.append(i)
fill('bars_confidence')

#Bar and beats_start
for i in range(len(dataset)):
    try:    
        mean('bars_start',i)
    except:
        count.append(i)
    try:    
        zero('beats_start',i)
        mean('beats_start',i)
    except:
        beats.append(i)

for i in count:
    dot('bars_start',i)
    mean('bars_start',i)
for i in beats:
    dot('beats_start',i)
    mean('beats_start',i)

fill('beats_start')
fill('bars_start')

l2 = list(dataset[dataset['beats_start']==0].index)
l3 = list(dataset[dataset['beats_confidence'] ==0].index)

for i in l2:
    dataset['beats_start'][i] = np.mean(dataset['beats_start'])
for i in l3:
    dataset['beats_confidence'][i] = np.mean(dataset['beats_confidence'])

#Fill artist familiarity missing values
fill('artist_familiarity')

for i in range(len(dataset)):
    if dataset['year'][i]==0:
        dataset['year'][i] = np.nan
#Fill missing value in year
dataset['year'] = dataset['year'].fillna(method='ffill').fillna(method='bfill')

#Sections Confidence
for i in range(len(dataset)):
    dataset['sections_confidence'][i] = str(dataset['sections_confidence'][i]).replace("1. ","1.0 ")

count=[]
beats = []
for i in range(len(dataset)):
    try:    
        mean('sections_confidence',i)
    except:
        count.append(i)
    try:    
        zero('sections_start',i)
        mean('sections_start',i)
    except:
        beats.append(i)
        
fill('sections_start')
fill('sections_confidence')

#Segments 
for i in range(len(dataset)):
    try:    
        mean('segments_confidence',i)
        mean('segments_loudness_max',i)
    except:
        count.append(i)

for i in(count):
    dot('segments_confidence',i)
    dot('segments_loudness_max',i)
    try:    
        zero('segments_confidence',i)
        mean('segments_confidence',i)
        zero('segments_loudness_max',i)
        mean('segments_loudness_max',i)
    except:
        count.append(i)
        
fill('segments_confidence')
fill('segments_loudness_max')
    
for i in range(len(dataset)):
    try:    
        mean('segments_loudness_max_time',i)
    except:
        beats.append(i)
for i in(beats):
    dot('segments_loudness_max_time',i)
    try:
        zero('segments_loudness_max_time',i)
        mean('segments_loudness_max_time',i)
    except:
        count.append(i)
fill('segments_loudness_max_time')

for i in range(len(dataset)):
    try:    
        mean('segments_loudness_start',i)
    except:
        beats.append(i)
for i in(beats):
    dot('segments_loudness_start',i)
    try:
        zero('segments_loudness_start',i)
        mean('segments_loudness_start',i)
    except:
        count.append(i)
fill('segments_loudness_start')




count=[]
beats = []
for i in range(len(dataset)):
    try:    
        mean('segments_pitches',i)
    except:
        beats.append(i)
for i in(beats):
    dot('segments_pitches',i)
    try:
        zero('segments_pitches',i)
        mean('segments_pitches',i)
    except:
        beats.append(i)
fill('segments_pitches')


for i in range(len(dataset)):
    try:    
        mean('segments_start',i)
    except:
        beats.append(i)
for i in(beats):
    dot('segments_start',i)
    try:
        zero('segments_start',i)
        mean('segments_start',i)
    except:
        beats.append(i)
fill('segments_start')

"""
count=[]
beats = []
dataset['segments_timbre'] = dataset['segments_timbre'].astype(object)
for i in range(len(dataset)):
    try:    
        mean('segments_timbre',i)
    except:
        beats.append(i)
for i in(beats):
    dot('segments_timbre',i)
    try:
        zero('segments_timbre',i)
        mean('segments_timbre',i)
    except:
        beats.append(i)
fill('segments_timbre')
"""

count=[]
beats = []
dataset['tatums_confidence'] = dataset['tatums_confidence'].astype(object)
for i in range(len(dataset)):
    try:    
        mean('tatums_confidence',i)
    except:
        beats.append(i)
for i in(beats):
    dot('tatums_confidence',i)
    try:
        zero('tatums_confidence',i)
        mean('tatums_confidence',i)
    except:
        count.append(i)
fill('tatums_confidence')

count=[]
beats = []
dataset['tatums_start'] = dataset['tatums_start'].astype(object)
for i in range(len(dataset)):
    try:    
        mean('tatums_start',i)
    except:
        beats.append(i)
for i in(beats):
    dot('tatums_start',i)
    try:
        zero('tatums_start',i)
        mean('tatums_start',i)
    except:
        count.append(i)

l1 = list(dataset[dataset['tatums_start']==2].index)
for i in l1:
    dataset['tatums_start'][i] = np.NaN
    
fill('tatums_start')
dataset.to_csv('Million_song_dataset.csv')

"""
import pandas as pd
new = pd.read_csv("Million.csv")
dataset['song_hotttnesss'] = new['song_hotttnesss']
dataset['tatums_start'].value_counts()    
"""

 
