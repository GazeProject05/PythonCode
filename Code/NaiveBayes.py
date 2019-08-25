#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 22 22:19:53 2019

@author: madhusharma
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from scipy.stats import multivariate_normal as mn
import math as m
import operator

##---------------------------- import csv-------------------------------------------------------------##
def loadCSV(filename):
    data = pd.read_excel(filename)
    return data

##----------------------------- filter column names and remove null values-----------------------------##
def filterColumn(dataFramename , columnNameArr):
    df = loadCSV("1Proband1.xlsx")
    df = df[columnNameArr]
    df = df.dropna()
    return df
##------------------------------declare global variables----------------------------------------------##
prior_probability = {
   'Scanning':0,
   'Skimming':0,
   'Reading':0,
   'MediaView':0,
   'Unknown' :0
    }
emission_probability = {
       'Scanning' : {'Fixation': 0, 'Saccade': 0, 'Unclassified':  0},
       'Skimming' : {'Fixation': 0, 'Saccade':  0, 'Unclassified': 0},
       'Reading' :  {'Fixation': 0, 'Saccade':  0, 'Unclassified': 0},
       'MediaView' : {'Fixation': 0, 'Saccade':  0, 'Unclassified': 0},
       'Unknown' : {'Fixation': 0, 'Saccade':  0, 'Unclassified': 0}
    }

##------------------------------ calculate prior probabilities----------------------------------------##
def priorProbabilities(trainFrame):
    total_probab =0
    for x in trainFrame['StudioEvent']:
        if 'Unknown' in x:
            #probab_unknown = probab_unknown+1
            total_probab = total_probab+1
            prior_probability['Unknown'] += 1
        elif('Reading' in x):
            #probab_Reading = probab_Reading+1
            total_probab = total_probab+1
            prior_probability['Reading'] += 1
        elif('Skimming' in x):
            #probab_skimming = probab_skimming+1
            total_probab = total_probab+1
            prior_probability['Skimming'] += 1
        elif('Scanning' in x):
            #probab_scanning = probab_scanning+1
            total_probab = total_probab+1
            prior_probability['Scanning'] += 1
        elif('MediaView' in x):
            #probab_mediaview = probab_mediaview+1
            total_probab = total_probab+1
            prior_probability['MediaView'] += 1
    prior_probability['Unknown'] = m.log(prior_probability['Unknown']) - m.log(total_probab)
    prior_probability['Reading'] = m.log(prior_probability['Reading']) - m.log(total_probab) 
    prior_probability['Skimming'] = m.log(prior_probability['Skimming']) - m.log(total_probab) 
    prior_probability['Scanning'] = m.log(prior_probability['Scanning'])- m.log(total_probab) 
    prior_probability['MediaView'] = m.log(prior_probability['MediaView']) - m.log(total_probab)
    return prior_probability

##------------------------------calculate the conditional probablity of GazeEventType-----------------------------##
def condionalProbability(train):
    total_Scan = 0
    total_Skim = 0
    total_Read = 0
    total_Unknown = 0
    total_Mediaview = 0
    for index, row in train.iterrows():
        if('Scanning' in row['StudioEvent'] and row['GazeEventType']== 'Fixation'):
            emission_probability['Scanning']['Fixation'] +=1
            total_Scan +=1
        elif('Scanning' in row['StudioEvent'] and row['GazeEventType']== 'Saccade'):
            emission_probability['Scanning']['Saccade'] +=1
            total_Scan +=1
        elif('Scanning' in row['StudioEvent'] and row['GazeEventType']== 'Unclassified'):
            emission_probability['Scanning']['Unclassified'] +=1
            total_Scan +=1
        elif('Skimming' in row['StudioEvent'] and row['GazeEventType']== 'Fixation'):
            emission_probability['Skimming']['Fixation'] +=1
            total_Skim +=1
        elif('Skimming' in row['StudioEvent'] and row['GazeEventType']== 'Saccade'):
            emission_probability['Skimming']['Saccade'] +=1
            total_Skim +=1
        elif('Skimming' in row['StudioEvent'] and row['GazeEventType']== 'Unclassified'):
            emission_probability['Skimming']['Unclassified'] +=1
            total_Skim +=1
        elif('Reading' in row['StudioEvent'] and row['GazeEventType']== 'Fixation'):
            emission_probability['Reading']['Fixation'] +=1
            total_Read +=1
        elif('Reading' in row['StudioEvent'] and row['GazeEventType']== 'Saccade'):
            emission_probability['Reading']['Saccade'] +=1
            total_Read +=1
        elif('Reading' in row['StudioEvent'] and row['GazeEventType']== 'Unclassified'):
            emission_probability['Reading']['Unclassified'] +=1
            total_Read +=1
        elif('MediaView' in row['StudioEvent'] and row['GazeEventType']== 'Fixation'):
            emission_probability['MediaView']['Fixation'] +=1
            total_Mediaview +=1
        elif('MediaView' in row['StudioEvent'] and row['GazeEventType']== 'Saccade'):
            emission_probability['MediaView']['Saccade'] +=1
            total_Mediaview +=1
        elif('MediaView' in row['StudioEvent'] and row['GazeEventType']== 'Unclassified'):
            emission_probability['MediaView']['Unclassified'] +=1
            total_Mediaview +=1
        elif('Unknown' in row['StudioEvent'] and row['GazeEventType']== 'Fixation'):
            emission_probability['Unknown']['Fixation'] +=1
            total_Unknown +=1
        elif('Unknown' in row['StudioEvent'] and row['GazeEventType']== 'Saccade'):
            emission_probability['Unknown']['Saccade'] +=1
            total_Unknown +=1
        elif('Unknown' in row['StudioEvent'] and row['GazeEventType']== 'Unclassified'):
            emission_probability['Unknown']['Unclassified'] +=1
            total_Unknown +=1
    #print("SCAN {} SKIM {} MEDIAVIEW {} READ {} UNKNOWN {}".format(total_Scan,total_Skim,total_Mediaview,total_Read,total_Unknown))
    emission_probability['Scanning']['Fixation'] = m.log(emission_probability['Scanning']['Fixation'])-m.log(total_Scan)
    emission_probability['Scanning']['Saccade'] = m.log(emission_probability['Scanning']['Saccade'])-m.log(total_Scan)
    emission_probability['Scanning']['Unclassified'] = m.log(emission_probability['Scanning']['Unclassified']) -m.log(total_Scan)
    emission_probability['Skimming']['Fixation'] =m.log(emission_probability['Skimming']['Fixation'] )-m.log (total_Skim)
    emission_probability['Skimming']['Saccade'] =m.log(emission_probability['Skimming']['Saccade'])-m.log(total_Skim)
    emission_probability['Skimming']['Unclassified'] =m.log(emission_probability['Skimming']['Unclassified'])-m.log(total_Skim)
    emission_probability['Reading']['Fixation'] =m.log( emission_probability['Reading']['Fixation'])- m.log(total_Read)
    emission_probability['Reading']['Saccade'] =m.log(emission_probability['Reading']['Saccade'])-m.log(total_Read)
    emission_probability['Reading']['Unclassified'] =m.log(emission_probability['Reading']['Unclassified'])-m.log(total_Read)
    emission_probability['MediaView']['Fixation'] =m.log(emission_probability['MediaView']['Fixation'])-m.log(total_Mediaview)
    emission_probability['MediaView']['Saccade'] =m.log(emission_probability['MediaView']['Saccade'])-m.log(total_Mediaview)
    emission_probability['MediaView']['Unclassified'] = m.log(emission_probability['MediaView']['Unclassified'])-m.log(total_Mediaview)
    emission_probability['Unknown']['Fixation'] =m.log(emission_probability['Unknown']['Fixation'])-m.log(total_Unknown)
    emission_probability['Unknown']['Saccade'] =m.log(emission_probability['Unknown']['Saccade'])-m.log(total_Unknown)
    emission_probability['Unknown']['Unclassified'] =m.log(emission_probability['Unknown']['Unclassified'])-m.log(total_Unknown)
    
    return emission_probability

##--------------------------------- Model for left pupil data-----------------------------------------------------## 
PupilModel = {
 'Reading' : (  {'mean': 2.4, 'std_dev': 0.11, 'weight': 0.2},
                 {'mean': 3.0, 'std_dev': 0.22, 'weight': 0.73},
                 {'mean': 3.8, 'std_dev': 0.16, 'weight': 0.064}
                ),
 'Scanning' : (  {'mean': 2.8, 'std_dev': 0.17, 'weight': 0.61},
                 {'mean': 3.9, 'std_dev': 0.35, 'weight': 0.13},
                 {'mean': 2.2, 'std_dev': 0.19, 'weight': 0.26}
                ),
 'Skimming' : (  {'mean': 2.6, 'std_dev': 0.3, 'weight': 0.22},
                 {'mean': 3.7, 'std_dev': 0.32, 'weight': 0.24},
                 {'mean': 3.0, 'std_dev': 0.19, 'weight': 0.55}
                ),
 'Unknown' : ({'mean': 3.0, 'std_dev': 0.4, 'weight': 1.0}, ),
 'MediaView' : (  {'mean': 2.7, 'std_dev': 0.29, 'weight': 0.75},
                 {'mean': 3.8, 'std_dev': 0.25, 'weight': 0.25}
                )

}

def gmmProbability(x, key):
    n = len(PupilModel[key])
    p = 0
    tempProbabs = []
    if(np.isnan(x)):              #checking for 0 (NAN) values
        return 0

    else:
        for i in range(n):
            tempProbabs.append( normalProbability(x, PupilModel[key][i]['mean'] , PupilModel[key][i]['std_dev'] )+ m.log(PupilModel[key][i]['weight'])  )
    p = logExpSum(tempProbabs)  
    return p


##-------------------------------Common functions------------------------------------------------------------------##
def findMaxArray(arr):
        maxValue = arr[0]
        for i in range(0, len(arr)):
            if(maxValue <arr[i]):
                maxValue = arr[i]
        return maxValue


def logExpSum(arr ):
        #find maximum of array assuming the array passed is already containg log values
        maxVal =0
        maxVal= findMaxArray(arr)
        res = 0
        for i in range(0, len(arr)):
            res += m.exp (arr[i] - maxVal) 
        return (m.log(res)+ maxVal)
    
def normalProbability(x, mean, std_dev):
    return ( (1/(std_dev*2.507)) * m.exp((-0.5)*m.pow( (x - mean)/std_dev , 2) ) )

def logPdf(datapoint, mean,deviation):
        #print("Calculating PDF")
        #u = (datapoint - self.mean) / abs(self.deviation)
        #y = -math.log(math.sqrt(2*math.pi*self.deviation * self.deviation))- (u*u/2)
        u = (datapoint - mean)
        y = -m.log(m.sqrt(2*m.pi*deviation))- (u*u/(2*deviation))
        #print("PDF: {} ".format(y))
        return y

##--------------------------------- GAZE GRADIENT'S MUTIVARIATE GAUSSIAN MODEL -----------------------------------##


ReadingMean = np.array([0.07644184938036225, 0.11914621067683508])
ReadingCov = np.array( [ [2515.59192225,  -19.91356905], [ -19.91356905,  436.1699851 ] ] )
ScanningMean = np.array([0.12469486264676968, 0.04143240586305546])
ScanningCov = np.array( [ [1818.29845049,  -29.88062487], [ -29.88062487,   86.47609229] ] )
SkimmingMean = np.array([0.323772999877606, -0.6836522377707968])
SkimmingCov = np.array( [ [2171.51980977,  -26.02772484], [ -26.02772484, 2033.8265622 ] ] )
UnknownMean = np.array([-0.12107822811927715, 0.08579523858885318])
UnknownCov = np.array( [ [1127.7907925 ,  149.01897013], [ 149.01897013,  599.13653165] ] )
MediaViewMean = np.array([-0.39917554105118513, 0.06389556853315012])
MediaViewCov = np.array( [ [689.83922956,  54.92173424], [ 54.92173424, 613.63371664] ] )

MultiVariateModel = {
'Reading' :     { 'MeanArray' : ReadingMean , 'Coovariance' : ReadingCov },
'Scanning':     { 'MeanArray' : ScanningMean , 'Coovariance' : ScanningCov },
'Skimming':     { 'MeanArray' : SkimmingMean , 'Coovariance' : SkimmingCov },
'Unknown' :     { 'MeanArray' : UnknownMean , 'Coovariance' : UnknownCov },
'MediaView':    { 'MeanArray' : MediaViewMean , 'Coovariance' : MediaViewCov },
}


# TO CALCULATE PROBAILITY FOR A GIVEN POINT,
#WE USE pdf FUNCTION GIVEN IN multivariate_normal CLASS FROM LIBRARY scipy.stats
# mn.pdf(x,mean,cov)

def mulnor(x, key):
    return mn.logpdf(x, mean = MultiVariateModel[key]['MeanArray'], cov = MultiVariateModel[key]['Coovariance'])
    
##-------------------------- Calculate posterior probabilities -------------------------------------##


def trainModel(columnNeeded):
    print("TRAIN MODEL")
    #load data 
    data = loadCSV("1Proband1.xlsx")
    # remove nan values and filter columns
    #columnNeeded = ['GazeEventType','PupilLeft','PupilRight','GazePointX','GazePointY','StudioEvent']
    df = filterColumn(data,columnNeeded)
    # split into train and test in ratio 80-20
    train, test = train_test_split(df, test_size=0.2)
    #calculate prior probabilities of class 
    prior_probability = priorProbabilities(train)
    #calculate conditional probabilities
    emission_probability = condionalProbability(train)
    #print(emission_probability)
    
def testModel(val):
    print("TEST MODEL")
    # calculate the probability for pupil data
    classPredicted = {
   'Scanning':0,
   'Skimming':0,
   'Reading':0,
   'MediaView':0,
   'Unknown' :0
    }
    #2d array columns for reading type rows for feature
    #predictedProbabilities = [4][5]
    w, h = 3, 4;
    predictedProbabilities = [[0 for x in range(w)] for y in range(h)] 
    # find the class of val 
    #1. calculate for left pupil data
    pupilProbabs = []
    for key in classPredicted:
        probab = gmmProbability(3.15, key)
        pupilProbabs.append(probab)
    predictedProbabilities[0] = pupilProbabs
        
    #2. find conditional probability of studioevent
    emissionProbabs = []
    for key in classPredicted:
        probab = emission_probability[key][val[0]]
        emissionProbabs.append(probab)
    predictedProbabilities[1]= emissionProbabs
    
    #3. calculate multinormal probability of gazegradient
    gazeProbabs = []
    for key in classPredicted:
        gazePts = [1829 , 25]
        probab = mulnor(gazePts, key)
        gazeProbabs.append(probab)
    predictedProbabilities[2] = gazeProbabs
    
    #4. multiply by prior probabilities
    priorProbabs = []
    for key in classPredicted:
        probab = prior_probability[key]
        priorProbabs.append(probab)
    predictedProbabilities[3]= priorProbabs
        
    print("PREDICTED PROBABILITIES {}".format(predictedProbabilities))

    #multiply probabilities of each feature for all classes
    #Scanning 
    tempScanProbab = []
    for i in range(0,4):
        tempScanProbab.append(predictedProbabilities[i][0])
    classPredicted['Scanning'] = logExpSum(tempScanProbab)
    
    #Skimming
    tempSkimProbab = []
    for i in range(0,4):
        tempSkimProbab.append(predictedProbabilities[i][1])
    classPredicted['Skimming'] = logExpSum(tempSkimProbab)
    
    #Reading
    tempReadProbab = []
    for i in range(0,4):
        tempReadProbab.append(predictedProbabilities[i][2])
    classPredicted['Reading'] = logExpSum(tempReadProbab)
    
    #MediaView
    tempMediaViewProbab = []
    for i in range(0,4):
        tempMediaViewProbab.append(predictedProbabilities[i][3])
    classPredicted['MediaView'] = logExpSum(tempMediaViewProbab)
    
    #Unknown
    tempUnknownProbab = []
    for i in range(0,4):
        tempUnknownProbab.append(predictedProbabilities[i][4])
    classPredicted['Unknown'] = logExpSum(tempUnknownProbab)
    print(classPredicted)  
    
    
    return max(classPredicted.items(), key=operator.itemgetter(1))[0]
    
    
        
    
    
    
trainModel(['GazeEventType','PupilLeft','PupilRight','GazePointX','GazePointY','StudioEvent']) 
print ("CLASS OF THE INSTANCE {}".format(testModel(['Fixation' , 3.15 , 1829 , 25 ])))
    

    
