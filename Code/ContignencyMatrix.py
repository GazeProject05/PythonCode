#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 17 11:24:54 2019

@author: madhusharma
"""
import pandas as pd
import numpy as np

import warnings
warnings.filterwarnings("ignore")
    
#filename = 'output_1st.csv'          #<-----------------------
filename = 'output_2nd.csv'

data = pd.read_csv(filename) 

data.head()
y_pred = data ['Prediction']
y_true_A = data['A']
y_true_B = data ['B']

y_predA = data[['Prediction','A']]
y_predB = data[['Prediction','B']]
gd = data[['A','B']]

def returnIndex(readingType):
    if readingType.lower() == 'skimming':
        return 0
    elif(readingType.lower() == 'scanning'):
        return 1
    elif(readingType.lower() == 'reading'):
        return 2
    elif(readingType.lower() == 'unknown'):
        return 3
    elif(readingType.lower() == 'mediaview'):
        return 4


w, h = 5, 5;


from sklearn import metrics

# Print the confusion matrix

#print("A wrt B, checking ground trurths against each other.")

print("CONFUSION MATRIX WRT B")     #<-----------------------



ConfusionMatrix = [[0 for x in range(w)] for y in range(h)]

for index, row in y_predB.iterrows():         #<------------- 
#for index, row in gd.iterrows(): 

    x = returnIndex(row['Prediction'])
#    x = returnIndex(row['A'])
    y = returnIndex(row['B'])                 #<----------------

    ConfusionMatrix[x][y] += 1

for i in range(0,5):
    print(ConfusionMatrix[i])
print('\n')    

accuratePredictions = 0
totalPredictions = 0
for i in range(0, len(ConfusionMatrix)):
    for j in range(0, len(ConfusionMatrix[0])):
        if(i==j):
            accuratePredictions += ConfusionMatrix[i][j]
            
        else:
            totalPredictions += ConfusionMatrix[i][j]
print("#ACCURATE PREDICTIONS {}".format(accuratePredictions))
print("#Incorrect PREDICTIONS {}".format(totalPredictions))
finalAccuracy = accuratePredictions/(totalPredictions+accuratePredictions)
print("Accuracy:", finalAccuracy)
print('\n')

# Print the precision and recall, among other metrics
                                            #<-----------------------------
print(metrics.classification_report(y_true_B, y_pred, digits=3))  


#------------------------


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
    x = np.array([num_skim,num_scan,num_read,num_unknown,num_media]) 
    print("majority class %:", np.max(x)/total)   
    print('\n')
    
    
print("PREDICTED VALUES")
findDistribution(y_pred)
print("ANNOTATION A")
findDistribution(y_true_A)
print("ANNOTATION B")
findDistribution(y_true_B)

       