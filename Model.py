# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 11:25:37 2019

@author: Ritwik Gupta
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score,f1_score

dataset = pd.read_csv('Million_song_dataset.csv')
features = pd.read_csv('Features_List.csv').drop('Unnamed: 0',axis=1)
labels = dataset['song_hotttnesss']
scores = []

#Splitting labels and features
features_train,features_test,labels_train,labels_test = train_test_split(features,labels,test_size = 0.2)

#Handling Data imbalance using Smote
from imblearn.over_sampling import SMOTE
sm = SMOTE(random_state=2)
features_train, labels_train = sm.fit_sample(features_train, labels_train.ravel())


#Scaling the features
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()  
features_train = sc.fit_transform(features_train)  
features_test = sc.transform(features_test)  

#XGB approach
from xgboost import XGBClassifier
model = XGBClassifier(booster = 'gbtree',eta=2)
model.fit(features_train,labels_train)
labels_pred = model.predict(features_test)

# evaluate predictions
accuracy = accuracy_score(labels_test,labels_pred)
print("Accuracy: %.2f%%" % (accuracy * 100.0))
scores.append(accuracy * 100.0)

#Checking If data imbalance is resolved
labels_pred_1 = pd.DataFrame(labels_pred)[0].value_counts()

#Applying Cross validation
from sklearn.model_selection import cross_val_score
score = cross_val_score(model,features_test,labels_test, cv=10)
print("Accuracy: %0.2f (+/- %0.2f)" % (score.mean(), score.std() * 2))
roc_auc_score(labels_test,labels_pred)
f1_score(labels_test, labels_pred, average='binary')
#Creating Pickle file
import pickle
with open('model','wb') as f:
    pickle.dump(model,f)    

"""
#ANN approach
l1=[]
from keras.models import Sequential
from keras.layers import Dense
classifier = Sequential()
classifier.add(Dense(units=17,kernel_initializer='uniform',activation='relu',input_dim=28))
classifier.add(Dense(units=17,kernel_initializer='uniform',activation='relu'))
classifier.add(Dense(units=1,kernel_initializer='uniform',activation='sigmoid'))
classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
classifier.fit(features_train,labels_train,batch_size=32,epochs=100)
labels_pred = classifier.predict(features_test)
for i in range(2000):
    if labels_pred[i]>0.5:
        labels_pred[i] = 1
    else:
        labels_pred[i] = 0
roc_auc_score(labels_test,labels_pred)
accuracy = accuracy_score(labels_test,labels_pred)
l1.append(accuracy)
print("Accuracy: %.2f%%" % (accuracy * 100.0))    

#Cross val score
if __name__== "__main__":
    from keras.wrappers.scikit_learn  import KerasClassifier
    from sklearn.model_selection import cross_val_score
    from keras.models import Sequential
    from keras.layers import Dense
    from keras import optimizers
    import numpy as np
    def build_classifier():
        classifier = Sequential()
        classifier.add(Dense(units=17,kernel_initializer='uniform',activation='relu',input_dim=13))
        classifier.add(Dense(units=17,kernel_initializer='uniform',activation='relu'))
        classifier.add(Dense(units=1,kernel_initializer='uniform',activation='sigmoid'))
        classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
        return classifier
    classifier = KerasClassifier(build_fn = build_classifier,batch_size=20,epochs=100)
    accuracies = cross_val_score(estimator = classifier,X=features_train,y=labels_train,cv = 10)
    mean = np.mean(accuracies)
    variance = accuracies.std()


#Testing params
from keras.wrappers.scikit_learn  import KerasClassifier
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense
def build_classifier(optimizer):
    classifier = Sequential()
    classifier.add(Dense(units=12,kernel_initializer='uniform',activation='relu',input_dim=18))
    classifier.add(Dense(units=12,kernel_initializer='uniform',activation='relu'))
    classifier.add(Dense(units=12,kernel_initializer='uniform',activation='relu'))
    classifier.add(Dense(units=1,kernel_initializer='uniform',activation='sigmoid'))
    classifier.compile(optimizer=optimizer,loss="binary_crossentropy",metrics=['accuracy'])
    return classifier
classifier = KerasClassifier(build_fn = build_classifier)
param = {'batch_size':[15,16,20,32,64],'nb_epoch':[50, 75, 100],'optimizer': ['adam','rmsprop']}
grid_search = GridSearchCV(estimator=classifier,param_grid=param,scoring='accuracy',cv=3)
grid_search = grid_search.fit(features_train,labels_train)
best_param = grid_search.best_params_
best_acc = grid_search.best_score_
print("Best: %f using %s" % (grid_search.best_score_, grid_search.best_params_))
means = grid_search.cv_results_['mean_test_score']
stds = grid_search.cv_results_['std_test_score']
params = grid_search.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
"""
"""
#DecisionTree approach

from sklearn.tree import DecisionTreeClassifier  
classifier = DecisionTreeClassifier()  
classifier.fit(features_train, labels_train)
labels_pred = classifier.predict(features_test)
roc_auc_score(labels_test,labels_pred)
f1_score(labels_test, labels_pred, average='binary')
accuracy = accuracy_score(labels_test,labels_pred)
print("Accuracy: %.2f%%" % (accuracy * 100.0))
scores.append(accuracy * 100.0)
#RandomForest approach

from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=57, random_state=0)  
classifier.fit(features_train, labels_train)  
labels_pred = classifier.predict(features_test)
accuracy = accuracy_score(labels_test,labels_pred)
roc_auc_score(labels_test,labels_pred)
from sklearn.metrics import f1_score
f1_score(labels_test, labels_pred, average='binary')
print("Accuracy: %.2f%%" % (accuracy * 100.0))
scores.append(accuracy * 100.0)

#SVM approach
from sklearn.svm import SVC
classifier = SVC(kernel = 'linear', random_state = 7)
classifier.fit(features_train, labels_train)
# Predicting the Test set results
labels_pred = classifier.predict(features_test)
roc_auc_score(labels_test,labels_pred)
f1_score(labels_test, labels_pred, average='binary')
accuracy = accuracy_score(labels_test,labels_pred)
print("Accuracy: %.2f%%" % (accuracy * 100.0))
scores.append(accuracy * 100.0)

#NaiveBayes approach
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
#bnb = GaussianNB()
bnb = BernoulliNB()
bnb.fit(features_train,labels_train)
labels_pred = bnb.predict(features_test)
roc_auc_score(labels_test,labels_pred)
f1_score(labels_test, labels_pred, average='binary')
accuracy = accuracy_score(labels_test,labels_pred)
print("Accuracy: %.2f%%" % (accuracy * 100.0))
scores.append(accuracy * 100.0)

#LogisticRegression
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(features_train, labels_train)
# Predicting the class labels
labels_pred = classifier.predict(features_test)
f1_score(labels_test, labels_pred, average='binary')
accuracy = accuracy_score(labels_test,labels_pred)
print("Accuracy: %.2f%%" % (accuracy * 100.0))
scores.append(accuracy * 100.0)

#K-nn Approach
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 56, p = 2)
classifier.fit(features_train, labels_train)
# Predicting the class labels
labels_pred = classifier.predict(features_test)
f1_score(labels_test, labels_pred, average='binary')
accuracy = accuracy_score(labels_test,labels_pred)
print("Accuracy: %.2f%%" % (accuracy * 100.0))
scores.append(accuracy * 100.0)

#Plotting the scores
approach = ['XGB','DecisionTree','RandomForest','SVM','NB','LR','KNN']
import matplotlib.pyplot as plt
plt.bar(approach,scores)

"""



