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

#print(data)

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

#------- Distribution of classes wrt A and B annotation and in predicted data
def findDistribution(arr):
    num_skim =0
    num_scan =0
    num_read =0
    num_media=0
    num_unknown=0
    total =0
    for i in range(0,len(arr)):
        if('Unknown' in arr[i]):
            num_unknown +=1
            total +=1
        elif('MediaView' in arr[i]):
            num_media +=1
            total +=1
        elif('Skimming' in arr[i]):
            num_skim +=1
            total +=1
        elif('Scanning' in arr[i]):
            num_scan +=1
            total +=1
        elif('Read' in arr[i]):
            num_read +=1
            total +=1
    print("SKIM:{} SCAN:{} READ:{} UNKNOWN:{} MEDIAVIEW:{} TOTAL:{}".format(num_skim,num_scan,num_read,num_unknown,num_media,total))
    
print("PREDICTED VALUES")
findDistribution(y_pred)
print("ANNOTATION A")
findDistribution(y_true_A)
print("ANNOTATION B")
findDistribution(y_true_B)

       