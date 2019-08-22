#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 17 11:24:54 2019

@author: madhusharma
"""
import pandas as pd
    
filename = 'output.csv'
data = pd.read_csv(filename) 
# Preview the first 5 lines of the loaded data 
data.head()
y_pred = data ['Prediction']
y_true_A = data['A']
y_true_B = data ['B']



from sklearn import metrics

# Print the confusion matrix
print("CONFUSION MATRIX WRT B")
ConfusionMatrix = metrics.confusion_matrix(y_true_B, y_pred,labels=["Unknown", "Mediaview", "Skimming", "Scanning", "Reading"])
accuratePredictions = 0
totalPredictions = 0
for i in range(0, len(ConfusionMatrix)):
    for j in range(0, len(ConfusionMatrix[0])):
        if(i==j):
            accuratePredictions += ConfusionMatrix[i][j]
            
        else:
            totalPredictions += ConfusionMatrix[i][j]
print("ACCURATE PREDICTIONS {}".format(accuratePredictions))
print("PREDICTIONS {}".format(totalPredictions))
finalAccuracy = accuratePredictions/(totalPredictions+accuratePredictions)
print(finalAccuracy)


# Print the precision and recall, among other metrics
print(metrics.classification_report(y_true_B, y_pred, digits=3))