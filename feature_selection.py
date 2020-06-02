#Importing the dataset using Pandas
import pandas as pd
import numpy as np
dataset = pd.read_csv('Datasets/Million_final.csv').drop('Unnamed: 0',axis=1)

#Checking for null values in the dataset
l1 = {}
for item in dataset:
    l1[item] = dataset[item].isnull().value_counts()

features = pd.read_csv('Datasets/Feature_List.csv').drop('Unnamed: 0',axis=1)
features.info()
labels = dataset['song_hotttnesss']

# Scale the features in the range 0-1
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()  
scaled_f = sc.fit_transform(features)  
features = pd.DataFrame(scaled_f,columns=features.columns)

#Feature Selection process
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV
import matplotlib.pyplot as plt

#Detect the optimal number of features
rfe_selector = RFECV(estimator=RandomForestClassifier(random_state=101), cv=StratifiedKFold(10), scoring='accuracy')
rfe_selector.fit(features, labels)
print('Optimal number of features: {}'.format(rfe_selector.n_features_))

#Plot the no. of features and their accuracy
opt = plt.figure(figsize=(16, 9))
plt.title('Optimal Feature Detection', fontsize=18, fontweight='bold', pad=20)
plt.xlabel('Number of features', fontsize=14, labelpad=20)
plt.ylabel('Correct Classification Score', fontsize=14, labelpad=20)
plt.plot(range(1, len(rfe_selector.grid_scores_) + 1), rfe_selector.grid_scores_, color='#303F9F', linewidth=3)
plt.show()
opt.savefig('Optimal_features.png')

#Calculate the list of features
rfe_support = rfe_selector.get_support()
rfe_feature = features.loc[:,rfe_support].columns.tolist()
print(str(len(rfe_feature)), 'selected features')

#Creating a wordcloud of the selected features
from wordcloud import WordCloud, STOPWORDS 
stopwords = set(STOPWORDS)
wc_features = ' '.join([str(elem) for elem in rfe_feature]) 
wordcloud = WordCloud(width = 800, height = 800,background_color ='white',min_font_size = 10,stopwords=stopwords).generate(wc_features)

wc = plt.figure(figsize = (8, 8), facecolor = None) 
plt.imshow(wordcloud) 
plt.axis("off") 
plt.tight_layout(pad = 0)   
plt.show() 
wc.savefig("Wordcloud.png")

#Saving the modified features and the dataset
features = features.loc[:,rfe_feature]
features.to_csv('Datasets/Feature_List.csv')
