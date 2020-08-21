import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import confusion_matrix
from sklearn import model_selection

#%matplotlib inline

import os


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
data = pd.read_csv("C:\\Users\\dell\\Downloads\\heart.csv")

from sklearn import model_selection

data.replace('?',0, inplace=True)

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
data = pd.concat([data]*3, ignore_index=True)

values = data.values

# Now impute it
imputer = SimpleImputer()
imputedData = imputer.fit_transform(values)

scaler = MinMaxScaler(feature_range=(0, 1))
normalizedData = scaler.fit_transform(imputedData)



from sklearn.metrics import confusion_matrix

X = normalizedData[:,0:13]
Y = normalizedData[:,13]


kfold = model_selection.KFold(n_splits=10,shuffle=True, random_state=7)

models = [KNeighborsClassifier(n_neighbors=8), DecisionTreeClassifier(max_depth=3, random_state=0), LogisticRegression(), 
        GaussianNB(), RandomForestClassifier(n_estimators=100, random_state=0)]
classifiers = [' KNN ', ' Decision Trees ', ' Logistic Regression ', ' Naive Bayes ', ' Random Forests ']
j=0
for i in models:
    print("Algorith = "+classifiers[j]+"\n")
    model=i
    results = model_selection.cross_val_score(model, X, Y, cv=kfold)
    print("Accuarcy: "+str(round(results.mean(),4)))
    print("Error = "+str((1-results.mean())*100))
    #print("\n -------------------------------------------------------\n")

    results = model_selection.cross_val_score(model, X, Y,scoring='average_precision', cv=kfold)
    print("Precision: "+str(round(results.mean(),4)))

    results = model_selection.cross_val_score(model, X, Y,scoring='f1', cv=kfold)
    print("F1 Score: "+str(round(results.mean(),4)))

    results = model_selection.cross_val_score(model, X, Y,scoring='recall', cv=kfold)
    print("ReCall Score: "+str(round(results.mean(),4)))
    results = model_selection.cross_val_score(model, X, Y,scoring='roc_auc', cv=kfold)
    print("ROC Value: "+str(round(results.mean(),4)))



    
    print("\n -------------------------------------------------------\n")

    j=j+1



from sklearn.ensemble import AdaBoostClassifier
seed = 7
num_trees = 100
kfold = model_selection.KFold(n_splits=10, random_state=seed,shuffle=True)
model = AdaBoostClassifier(n_estimators=num_trees, random_state=seed)
results = model_selection.cross_val_score(model, X, Y, cv=kfold)
print("Algorith =  AdaBoostClassifier")
print("Accuarcy: "+str(round(results.mean(),4)))
    #print("\n -------------------------------------------------------\n")

results = model_selection.cross_val_score(model, X, Y,scoring='average_precision', cv=kfold)
print("Precision: "+str(round(results.mean(),4)))
print("Error = "+str(1-results.mean()))

results = model_selection.cross_val_score(model, X, Y,scoring='f1', cv=kfold)
print("F1 Score: "+str(round(results.mean(),4)))

results = model_selection.cross_val_score(model, X, Y,scoring='recall', cv=kfold)
print("ReCall Score: "+str(round(results.mean(),4)))
results = model_selection.cross_val_score(model, X, Y,scoring='roc_auc', cv=kfold)
print("ROC Value: "+str(round(results.mean(),4)))
print("\n -------------------------------------------------------\n")

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import VotingClassifier

#kfold = model_selection.KFold(n_splits=10, random_state=seed)
# create the sub models
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

estimators = []
model1 = LogisticRegression()
estimators.append(('logistic', model1))
model2 = DecisionTreeClassifier()
estimators.append(('cart', model2))
model3 = SVC()
estimators.append(('svm', model3))
model4=GaussianNB()
estimators.append(('navi',model4))
# create the ensemble model
model = VotingClassifier(estimators)
print("Algorith =  Ensemble: Majorty Voting")
print("Accuarcy: "+str(round(results.mean(),4)))

    #print("\n -------------------------------------------------------\n")

'''results = model_selection.cross_val_score(model, X, Y,scoring='average_precision', cv=kfold)
print("Precision: "+str(round(results.mean(),4)))

results = model_selection.cross_val_score(model, X, Y,scoring='f1', cv=kfold)
print("F1 Score: "+str(round(results.mean(),4)))

results = model_selection.cross_val_score(model, X, Y,scoring='recall', cv=kfold)
print("ReCall Score: "+str(round(results.mean(),4)))
results = model_selection.cross_val_score(model, X, Y,scoring='roc_auc', cv=kfold)
print("ROC Value: "+str(round(results.mean(),4)))'''
print("\n -------------------------------------------------------\n")

cart = DecisionTreeClassifier()
num_trees = 100
model = BaggingClassifier(base_estimator=cart, n_estimators=num_trees, random_state=7)
print("Algorith =  Bagging Classifier")
print("Accuarcy: "+str(round(results.mean(),4)))
print("Error = "+str(1-results.mean()))
    #print("\n -------------------------------------------------------\n")

results = model_selection.cross_val_score(model, X, Y,scoring='average_precision', cv=kfold)
print("Precision: "+str(round(results.mean(),4)))

results = model_selection.cross_val_score(model, X, Y,scoring='f1', cv=kfold)
print("F1 Score: "+str(round(results.mean(),4)))

results = model_selection.cross_val_score(model, X, Y,scoring='recall', cv=kfold)
print("ReCall Score: "+str(round(results.mean(),4)))
results = model_selection.cross_val_score(model, X, Y,scoring='roc_auc', cv=kfold)
print("ROC Value: "+str(round(results.mean(),4)))
print("\n -------------------------------------------------------\n")