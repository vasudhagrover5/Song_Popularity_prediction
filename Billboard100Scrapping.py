# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 10:50:27 2019

@author: Ritwik Gupta
"""


##### Getting the data from billboard archive #####

import pandas as pd
import matplotlib.pyplot as plt
import requests
from bs4 import BeautifulSoup
from collections import OrderedDict
import os

col = ['Artist','Title','Year']
dataframe = pd.DataFrame()
for i in range(1947,2013):
    artist = []
    song_title = []
    year = []
    url = "http://billboardtop100of.com/"+str(i)+"-2/"
    source = requests.get(url).text
    soup = BeautifulSoup(source,'lxml')
    table = soup.find('alignleft',class_ = 'table')
    for row in soup.find_all('tr'):
        cells = row.find_all('td')
        if len(cells) == 3:
            artist.append(cells[1].text.strip())
            song_title.append(cells[2].text.strip())
    df = pd.DataFrame() 
    year = [i for k in range(0,len(artist))]
    col_data = OrderedDict(zip(col,[artist,song_title,year]))
    df = pd.DataFrame(col_data)
    dataframe = dataframe.append(df)
dataframe.encode('UTF-8')

os.chdir('Datasets')
dataframe.to_csv('Billboardtop100.csv')
    
