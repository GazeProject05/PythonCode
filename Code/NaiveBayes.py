import pandas as pd
from sklearn.model_selection import train_test_split
from scipy.stats import multivariate_normal as mn
import math as m
import operator
import numpy as np

##---------------------------- import csv-------------------------------------------------------------##
def loadCSV(filename):
    data = pd.read_excel(filename)
    return data

##----------------------------- filter column names and remove null values-----------------------------##
def filterColumn(dataFramename , columnNameArr):
    #df = loadCSV("NaiveBayes.xlsx")
    dataFramename = dataFramename[columnNameArr]
    dataFramename = dataFramename.dropna()
    return dataFramename
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
emission_probabilityB = {
       'Scanning' : {'Fixation': 0, 'Saccade': 0, 'Unclassified':  0},
       'Skimming' : {'Fixation': 0, 'Saccade':  0, 'Unclassified': 0},
       'Reading' :  {'Fixation': 0, 'Saccade':  0, 'Unclassified': 0},
       'MediaView' : {'Fixation': 0, 'Saccade':  0, 'Unclassified': 0},
       'Unknown' : {'Fixation': 0, 'Saccade':  0, 'Unclassified': 0}
    }
trainFrame = []
testFrame = []
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
 'Reading' : (  {'mean': 2.7, 'std_dev': 0.17, 'weight': 0.43},
                 {'mean': 3.3, 'std_dev': 0.35, 'weight': 0.39},
                 {'mean': 2.4, 'std_dev': 0.11, 'weight': 0.19}
                ),
 'Scanning' : (  {'mean': 2.8, 'std_dev': 0.14, 'weight': 0.43},
                 {'mean': 2.4, 'std_dev': 0.15, 'weight': 0.18},
                 {'mean': 3.2, 'std_dev': 0.18, 'weight': 0.38}
                ),
 'Skimming' : (   {'mean': 3.8, 'std_dev': 0.31, 'weight': 0.11},
                  {'mean': 2.8, 'std_dev': 0.32, 'weight': 0.89}
                ),
 'Unknown' : ({'mean': 2.9, 'std_dev': 0.52, 'weight': 1.0}, ),
 
 'MediaView' : (  {'mean': 2.8, 'std_dev': 0.37, 'weight': 0.29},
                  {'mean': 3.3, 'std_dev': 0.4, 'weight': 0.25},
                 {'mean': 2.7, 'std_dev': 0.12, 'weight': 0.46}
                )

}

def gmmProbability(x, key):
    n = len(PupilModel[key])
    p = 0
    tempProbabs = []
    #if(np.isnan(x)):              #checking for 0 (NAN) values
    #    return 0

    #else:
    for i in range(n):
        #print("X {}".format(x))
        #print("KEY {}".format(key))
        #print("Mean {}".format(PupilModel[key][i]['mean']))
        #print("STD {}".format(PupilModel[key][i]['std_dev']))
        #print("WEIGHT {}".format(PupilModel[key][i]['weight']))
        #print("X {} PupilModel[key][i]['mean'] {} PupilModel[key][i]['std_dev']{} PupilModel[key][i]['weight'] {}".format(x,PupilModel[key][i]['mean'],PupilModel[key][i]['std_dev'],PupilModel[key][i]['weight']))
        tempProbabs.append( logPdf(x, PupilModel[key][i]['mean'] , PupilModel[key][i]['std_dev'] )+ m.log(PupilModel[key][i]['weight'])  )
    p = logExpSum(tempProbabs)  
    return p
##--------------------------------- Model for right pupil data-----------------------------------------------------## 
PupilModelRight = {
 'Reading' : (  {'mean': 2.9, 'std_dev': 0.17, 'weight': 0.28},
                 {'mean': 2.4, 'std_dev': 0.16, 'weight': 0.38},
                 {'mean': 3.5, 'std_dev': 0.35, 'weight': 0.33}
                ),
 'Scanning' : (  {'mean': 2.4, 'std_dev': 0.2, 'weight': 0.33},
                 {'mean': 3.7, 'std_dev': 0.42, 'weight': 0.16},
                 {'mean': 2.9, 'std_dev': 0.23, 'weight': 0.51}
                ),
 'Skimming' : (  {'mean': 2.9, 'std_dev': 0.36, 'weight': 0.92},
                 {'mean': 4.1, 'std_dev': 0.29, 'weight': 0.079}
                ),
 'Unknown' : ({'mean': 2.9, 'std_dev': 0.53, 'weight': 1.0}, ),
    
 'MediaView' : (  {'mean': 2.6, 'std_dev': 0.18, 'weight': 0.5},
                  {'mean': 3.2, 'std_dev': 0.18, 'weight': 0.35},
                 {'mean': 3.7, 'std_dev': 0.3, 'weight': 0.15}
                )

}

def gmmProbabilityRight(x, key):
    
    #print("gmmProbabilityRight {} {}".format(x,key))
    n = len(PupilModelRight[key])
    p = 0
    tempProbabs = []
    #if(np.isnan(x)):              #checking for 0 (NAN) values
    #    return 0

    #else:
    #print("n {}".format(n))
    for i in range(n):
        #print("LOGPDF {}".format(logPdf(x, PupilModel[key][i]['mean'] , PupilModel[key][i]['std_dev'])))
        
        tempProbabs.append( logPdf(x, PupilModelRight[key][i]['mean'] , PupilModelRight[key][i]['std_dev'] )+ m.log(PupilModelRight[key][i]['weight']) ) 
    #print("TEMPPROBABS {}".format(tempProbabs))
    p = logExpSum(tempProbabs)  
    return p

##-------------------------------Common functions------------------------------------------------------------------##
def findMaxArray(arr):
    #print(arr)
    #print(type(arr))
    #print("LENGTH ARR {}".format(len(arr)))
    #print("LENGTH ARR[0] {}".format(len(arr[0])))
    maxValue = arr[0]
    #print("MAXVALUE {}".format(maxValue))
    for i in range(0, len(arr)):
        if(maxValue < arr[i]):
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



ReadingMean = np.array([-0.018223117030612773, 0.04327775196599728])
ReadingCov = np.array( [ [1179.63428727, 171.67930601], [171.67930601, 836.91103656] ] )

ScanningMean = np.array([-0.14131410896028737, -0.08072241069646777])
ScanningCov = np.array( [ [1852.15462508, 330.37926778], [330.37926778, 1716.64905786] ] )

SkimmingMean = np.array([0.3212, -0.6845777777777777])
SkimmingCov = np.array( [ [1805.68290536, 216.39954858], [216.39954858, 4072.86681401] ] )

UnknownMean = np.array([-0.04316383904262544, 0.06364754324031643])
UnknownCov = np.array( [ [1786.47885221, 598.16561454], [598.16561454, 2461.48454459] ] )

MediaViewMean = np.array([-1.0968448729184925, 0.2287467134092901])
MediaViewCov = np.array( [ [1922.47066957, -383.06551818], [-383.06551818, 1367.8950435] ] )




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
data = pd.read_excel("NaiveBayes.xlsx")
# remove nan values and filter columns
#columnNeeded = ['GazeEventType','PupilLeft','PupilRight','GazePointX','GazePointY','StudioEvent']
#print(columnNeeded)
#df = filterColumn(data,columnNeeded)
data = data[['GazeEventType','GazeEventType_B','PupilLeft','PupilRight','GazeGradientX','GazeGradientY','StudioEvent','StudioEvent_B']]
data = data.dropna()
#print(data)
# split into train and test in ratio 80-20
train,test = train_test_split(data, test_size=0.2)
#print("TRAIN")
#print(train)

def trainModel(columnNeeded):
    print("TRAIN MODEL")
    #calculate prior probabilities of class 
    prior_probability = priorProbabilities(train)
    #calculate conditional probabilities
    emission_probability = condionalProbability(train)
    #print(emission_probability)
    
def testModel(GazeEventType,PupilLeft,PupilRight,GazeGradientX,GazeGradientY):
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
    w, h = 5, 5;
    predictedProbabilities = [[0 for x in range(w)] for y in range(h)] 
    #print("DIMENSIONS OF predictedProbabilities len{} breadth{}".format(len(predictedProbabilities),len(predictedProbabilities[0])))
    #1.calculate for left pupil data
    pupilProbabs = []
    for key in classPredicted:
        probab = gmmProbability(PupilLeft, key)
        pupilProbabs.append(probab)
    predictedProbabilities[0] = pupilProbabs
    
    #2.calculate for right pupil data
    rightPupilProbabs = []
    for key in classPredicted:
        probab = gmmProbabilityRight(PupilRight, key)
        rightPupilProbabs.append(probab)
    predictedProbabilities[1] = rightPupilProbabs
        
    #3. find conditional probability of geze event type
    emissionProbabs = []
    for key in classPredicted:
        probab = emission_probability[key][GazeEventType]
        emissionProbabs.append(probab)
    predictedProbabilities[2]= emissionProbabs
    
    #4. calculate multinormal probability of gazegradientA
    gazeProbabs = []
    for key in classPredicted:
        gazePts = [GazeGradientX , GazeGradientY]
        probab = mulnor(gazePts, key)
        gazeProbabs.append(probab)
    predictedProbabilities[3] = gazeProbabs
    

    #5. multiply by prior probabilities
    priorProbabs = []
    for key in classPredicted:
        probab = prior_probability[key]
        priorProbabs.append(probab)
    predictedProbabilities[4]= priorProbabs
        
    #print("PREDICTED PROBABILITIES {}".format(predictedProbabilities))

    #multiply probabilities of each feature for all classes
    #Scanning 
    tempScanProbab = []
    for i in range(0,5):
        tempScanProbab.append(predictedProbabilities[i][0])
    classPredicted['Scanning'] = logExpSum(tempScanProbab)
    
    #Skimming
    tempSkimProbab = []
    for i in range(0,5):
        tempSkimProbab.append(predictedProbabilities[i][1])
    classPredicted['Skimming'] = logExpSum(tempSkimProbab)
    
    #Reading
    tempReadProbab = []
    for i in range(0,5):
        tempReadProbab.append(predictedProbabilities[i][2])
    classPredicted['Reading'] = logExpSum(tempReadProbab)
    
    #MediaView
    tempMediaViewProbab = []
    for i in range(0,5):
        tempMediaViewProbab.append(predictedProbabilities[i][3])
    classPredicted['MediaView'] = logExpSum(tempMediaViewProbab)
    
    #Unknown
    tempUnknownProbab = []
    for i in range(0,5):
        tempUnknownProbab.append(predictedProbabilities[i][4])
    classPredicted['Unknown'] = logExpSum(tempUnknownProbab)
    #print(classPredicted)  
    
    
    return max(classPredicted.items(), key=operator.itemgetter(1))[0]
    
    
        
    
  
    
trainModel(['GazeEventType','PupilLeft','PupilRight','GazeGradientX','GazeGradientY','StudioEvent']) 
fields = ['Prediction', 'A','B']
with open('ZeroOrderOutput.csv','w')as file:
    writer = csv.DictWriter(file,fieldnames = fields)
    writer.writeheader()
    for index, row in test.iterrows():
        if((row['StudioEvent'].lower() or row['StudioEvent_B'].lower()) == '0_Unstated'.lower()):
            continue
        else:
            #GazeEventTypeB = row['GazeEventType_B']
            GazeEventType = row['GazeEventType']
            PupilLeft = row['PupilLeft']
            PupilRight = row['PupilRight']
            GazeGradientX= row['GazeGradientX']
            GazeGradientY= row['GazeGradientY']
            #print (GazeGradientX, GazeGradientY)
            predicted_val = testModel(GazeEventType,PupilLeft,PupilRight,GazeGradientX,GazeGradientY)
            print ("{},{} PREDICTED {}".format(row['StudioEvent'],row['StudioEvent_B'],predicted_val))
            a= row['StudioEvent'].split('_')
            b=row['StudioEvent_B'].split('_')
            print("B {}".format(b))
            data = [{'Prediction':predicted_val,'A':a[1],'B':b[1]}]
            writer.writerows(data)
print("DONE")
    

    