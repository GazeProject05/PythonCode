import pandas as pd
import numpy as np
import math as m
from scipy.stats import multivariate_normal as mn
import csv

#Reading data file
df = pd.read_excel('19Proband19.xlsx')

#Readig relevant columns of data
gazeEventData = df['GazeEventType']

b = df['PupilLeft']
leftPupilData = b.str.replace(',','.').astype(float)
b2 = df['PupilRight']
rightPupilData = b2.str.replace(',','.').astype(float)

gazeGradientData = df[['GazeGradientX','GazeGradientY']]


#GroundTruth | Annotations made by expret A and B
gd1 = df['StudioEvent']
gd2 = df['StudioEvent_B']

                            #WHY are obs and y2 ARE THESE DIFFERENT??????
                            # y2 = df['GazeEventType_B']
                            # y3= df['GazeEventTypeDiff']


# Modeling
states = ('Scanning', 'Skimming', 'Reading', 'MediaView', 'Unknown')
start_probability = {'Scanning': 0.17857142857, 'Skimming': 0.0, 'Reading': 0.53571428571, 'MediaView': 0.0, 'Unknown':  0.28571428571}


#Dummy data for testing GazeEventType
observations = ('Fixation','Saccade','Saccade','Fixation','Fixation','Fixation','Fixation')


#-------------------- 2nd order Transition Matrix ----------#

transition = {
        'Scanning': {
                'Scanning' : {'Scanning': -0.00232644438500174, 'Skimming':-7.88618962102806, 'Reading': -7.53254958078448, 'MediaView': -9.24263101899826, 'Unknown': -6.63342665253269},
                'Skimming' : {'Scanning': 0.0, 'Skimming': 0.0, 'Reading': 0.0, 'MediaView': 0.0, 'Unknown': 0.0},
                'Reading' :  {'Scanning': 0.0, 'Skimming': 0.0, 'Reading': 0.0, 'MediaView': 0.0, 'Unknown': 0.0},
                'MediaView' :{'Scanning': 0.0, 'Skimming': 0.0, 'Reading': 0.0, 'MediaView': 0.0, 'Unknown': 0.0},
                'Unknown' :  {'Scanning': 0.0, 'Skimming': 0.0, 'Reading': 0.0, 'MediaView': 0.0, 'Unknown': 0.0}
                    },
        
        'Skimming': {
                'Scanning' : {'Scanning': 0.0, 'Skimming': 0.0, 'Reading': 0.0, 'MediaView': 0.0, 'Unknown': 0.0},
                'Skimming' : {'Scanning': -7.48754377378507, 'Skimming': -0.00237288416630221, 'Reading': -7.22243602337183, 'MediaView': -9.11500019172185, 'Unknown': -6.93818448601684},
                'Reading' :  {'Scanning': 0.0, 'Skimming': 0.0, 'Reading': 0.0, 'MediaView': 0.0, 'Unknown': 0.0},
                'MediaView' :{'Scanning': 0.0, 'Skimming': 0.0, 'Reading': 0.0, 'MediaView': 0.0, 'Unknown': 0.0},
                'Unknown' :  {'Scanning': 0.0, 'Skimming': 0.0, 'Reading': 0.0, 'MediaView': 0.0, 'Unknown': 0.0}
                    },
        
          'Reading': {
                'Scanning' : {'Scanning': 0.0, 'Skimming': 0.0, 'Reading': 0.0, 'MediaView': 0.0, 'Unknown': 0.0},
                'Skimming' : {'Scanning': 0.0, 'Skimming': 0.0, 'Reading': 0.0, 'MediaView': 0.0, 'Unknown': 0.0},
                'Reading' :  {'Scanning': -7.6817646949916, 'Skimming': -8.02586085672347, 'Reading': -0.00201595889106265, 'MediaView': -10.1053023984033, 'Unknown': -6.73800656841683},
                'MediaView' :{'Scanning': 0.0, 'Skimming': 0.0, 'Reading': 0.0, 'MediaView': 0.0, 'Unknown': 0.0},
                'Unknown' :  {'Scanning': 0.0, 'Skimming': 0.0, 'Reading': 0.0, 'MediaView': 0.0, 'Unknown': 0.0}
                    },
          
            'MediaView': {
                'Scanning' : {'Scanning': 0.0, 'Skimming': 0.0, 'Reading': 0.0, 'MediaView': 0.0, 'Unknown': 0.0},
                'Skimming' : {'Scanning': 0.0, 'Skimming': 0.0, 'Reading': 0.0, 'MediaView': 0.0, 'Unknown': 0.0},
                'Reading' :  {'Scanning': 0.0, 'Skimming': 0.0, 'Reading': 0.0, 'MediaView': 0.0, 'Unknown': 0.0},
                'MediaView' :{'Scanning': -6.93723317180174, 'Skimming': -7.02424454879137, 'Reading': -7.81270190915564, 'MediaView': -0.00372892779686396, 'Unknown': -6.53176806369357},
                'Unknown' :  {'Scanning': 0.0, 'Skimming': 0.0, 'Reading': 0.0, 'MediaView': 0.0, 'Unknown': 0.0}
                    },
            
              'Unknown': {
                'Scanning' : {'Scanning': 0.0, 'Skimming': 0.0, 'Reading': 0.0, 'MediaView': 0.0, 'Unknown': 0.0},
                'Skimming' : {'Scanning': 0.0, 'Skimming': 0.0, 'Reading': 0.0, 'MediaView': 0.0, 'Unknown': 0.0},
                'Reading' :  {'Scanning': 0.0, 'Skimming': 0.0, 'Reading': 0.0, 'MediaView': 0.0, 'Unknown': 0.0},
                'MediaView' :{'Scanning': 0.0, 'Skimming': 0.0, 'Reading': 0.0, 'MediaView': 0.0, 'Unknown': 0.0},
                'Unknown' :  {'Scanning': -7.50494623776424, 'Skimming': -8.39876411378634, 'Reading': -7.9629015284938, 'MediaView': -10.6830000681122, 'Unknown': -0.00114723164757713}
                    }
        
        }
              

#print(transition['Reading']['Reading']['Unknown']) 
# t-2 -> Reading -> state 2 time steps ago
# t-1 -> Reading -> state 1 time step ago
# t   -> Unkown -> state in this time step    
       
        
##------------------------------- MODEL FOR GAZE EVENT TYPE ---------------------------------------##

emission_probability = {
   'Scanning' : {'Fixation': 0.63248022558, 'Saccade': 0.28533467154, 'Unclassified':  0.08218510287},
   'Skimming' : {'Fixation': 0.54331139442, 'Saccade':  0.36026790895, 'Unclassified': 0.09642069662},
   'Reading' : {'Fixation': 0.76076073655, 'Saccade':  0.18806560675, 'Unclassified': 0.05117365669},
   'MediaView' : {'Fixation': 0.69896303332, 'Saccade':  0.2433309586, 'Unclassified': 0.05770600807},
   'Unknown' : {'Fixation': 0.42320040043, 'Saccade': 0.19158475227, 'Unclassified': 0.38521484729}
}



##--------------------    MODEL FOR PUPIL DATA -- AND -- FUNCTIONS TO CALCULATE PROBABILITIES  ------------------##

leftPupilModel = {
 'Reading' : (  {'mean': 3.8, 'std_dev': 0.19, 'weight': 0.1},
                 {'mean': 2.4, 'std_dev': 0.077, 'weight': 0.15},
                 {'mean': 2.8, 'std_dev': 0.28, 'weight': 0.75}
                ),
 'Scanning' : (  {'mean': 2.9, 'std_dev': 0.2, 'weight': 0.61},
                 {'mean': 3.9, 'std_dev': 0.26, 'weight': 0.11},
                 {'mean': 2.4, 'std_dev': 0.21, 'weight': 0.28}
                ),
 'Skimming' : (  {'mean': 2.8, 'std_dev': 0.34, 'weight': 0.92},
                 {'mean': 3.9, 'std_dev': 0.24, 'weight': 0.083}
                ),
 'Unknown' : ({'mean': 2.7, 'std_dev': 0.47, 'weight': 1.0}, ),
 'MediaView' : (  {'mean': 3.6, 'std_dev': 0.4, 'weight': 0.1},
                  {'mean': 2.8, 'std_dev': 0.33, 'weight': 0.42}, 
                  {'mean': 2.7, 'std_dev': 0.12, 'weight': 0.48}
                )

}


rightPupilModel = {
 'Reading' : (  {'mean': 2.9, 'std_dev': 0.27, 'weight': 0.71},
                 {'mean': 2.2, 'std_dev': 0.071, 'weight': 0.18},
                 {'mean': 3.9, 'std_dev': 0.19, 'weight': 0.11}
                ),
 'Scanning' : (  {'mean': 2.2, 'std_dev': 0.11, 'weight': 0.19},
                 {'mean': 4.1, 'std_dev': 0.25, 'weight': 0.1},
                 {'mean': 2.9, 'std_dev': 0.24, 'weight': 0.7}
                ),
 'Skimming' : (  {'mean': 4.2, 'std_dev': 0.25, 'weight': 0.072},
                 {'mean': 2.9, 'std_dev': 0.37, 'weight': 0.93}
                ),
 'Unknown' : ({'mean': 2.8, 'std_dev': 0.49, 'weight': 1.0}, ),
 'MediaView' : (  {'mean': 2.6, 'std_dev': 0.18, 'weight': 0.57},
                  {'mean': 3.7, 'std_dev': 0.29, 'weight': 0.15},
                  {'mean': 3.1, 'std_dev': 0.13, 'weight': 0.28}
                )

}


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


def gmmProbability(x, key, side):
    
    p=0
    tempProbabs = []
    
    #if(np.isnan(x)):              #checking for 0 (NAN) values
    #    p = 0
    
    
    if(side=='left'):           #side -> decides which (left or right) pupil model are we going to use. 
        n = len(leftPupilModel[key])        
        for i in range(n):
            tempProbabs.append(logPdf(x, leftPupilModel[key][i]['mean'] , leftPupilModel[key][i]['std_dev'] )+ m.log(leftPupilModel[key][i]['weight']))
    #        p = p + ( normalProbability(x, leftPupilModel[key][i]['mean'] , leftPupilModel[key][i]['std_dev'] )  * leftPupilModel[key][i]['weight']  )
        p = logExpSum(tempProbabs)
    
    elif(side=='right'):
        n = len(rightPupilModel[key])
        for i in range(n):
            tempProbabs.append(logPdf(x, rightPupilModel[key][i]['mean'] , rightPupilModel[key][i]['std_dev'] )+ m.log(rightPupilModel[key][i]['weight']))
    #        p = p + ( normalProbability(x, rightPupilModel[key][i]['mean'] , rightPupilModel[key][i]['std_dev'] )  * rightPupilModel[key][i]['weight']  )
        p = logExpSum(tempProbabs)
        
    return p



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
 #   if(np.isnan(x[0] or x[1]) ):     #checking for 0 (NAN) values
  #      return 0
  #  else:
  return mn.logpdf(x, mean = MultiVariateModel[key]['MeanArray'], cov = MultiVariateModel[key]['Coovariance'])
        
              
    
#--------------- Viterbi Implimentation -----------------#

# Helps visualize the steps of Viterbi.
def print_dptable(V):
    s = "    " + " ".join(("%7d" % i) for i in range(len(V))) + "\n"
    for y in V[0]:
        s += "%.5s: " % y
        s += " ".join("%.7s" % ("%f" % v[y]) for v in V)
        s += "\n"
    print(s)
    
    
#Viterbi algo function
def viterbi(gazeEventData, leftPupilData, rightPupilData, gazeGradientData, states, start_p, trans_p, emit_p):
    V = [{}]                        #[]-> List ; {} -> Dictionary        [{}] ->List of dictionar
    path = {}
    
    # Initialize base cases (t == 0)
    for y in states:
        #USE LOG HERE
        #array = [m.log1p(start_p[y]) , m.log1p(emit_p[y][gazeEventData[0]]) , m.log1p(gmmProbability(leftPupilData[0] ,y, 'left')) , m.log1p(gmmProbability(rightPupilData[0] ,y, 'right')) , m.log1p(mulnor(gazeGradientData.iloc[0], y))]
        array = []
        #1st ->  Logic for skipping probilities, when data is not presents
        
        if(start_p[y] != 0.0):
            array.append(m.log(start_p[y]))
                
        if(pd.isnull(gazeEventData[0]) == False):
            array.append(m.log(emit_p[y][gazeEventData[0]]))
                    
        if(pd.isnull(leftPupilData[0]) == False):    
            array.append(gmmProbability(leftPupilData[0], y, 'left'))
        if(pd.isnull(rightPupilData[0]) == False):    
            array.append(gmmProbability(rightPupilData[0], y, 'right'))
                
        if((gazeGradientData.iloc[0].dropna().empty) == False):
            array.append(mulnor(gazeGradientData.iloc[0], y))
        
        
        V[0][y] = logExpSum(array)  
        path[y] = [y]

   


    # Run Viterbi for (t >= 1)
    for t in range(1, len(gazeEventData)):
        V.append({})
        newpath = {}

        for y in states:
            maximum = float("-inf")
            state = ''
            array = []     
            
            for y0 in states:                
                x = (path[y0][t-2])
                #array = [ (V[t-1][y0]) , ( trans_p[x][y0][y] ) , m.log1p( emit_p[y][gazeEventData[t]] ) , m.log1p(gmmProbability(leftPupilData[t], y, 'left')) , m.log1p(gmmProbability(rightPupilData[t], y, 'right'))  , m.log1p(mulnor(gazeGradientData.iloc[t], y)) ]
                
                 #1st ->  Logic for skipping probilities, when data is not presents
                array = [ (V[t-1][y0]), trans_p[x][y0][y] ]
                
                if(pd.isnull(gazeEventData[t]) == False):
                    array.append(m.log(emit_p[y][gazeEventData[t]]))
                    
                if(pd.isnull(leftPupilData[t]) == False):    
                    array.append(gmmProbability(leftPupilData[t], y, 'left'))
                if(pd.isnull(rightPupilData[t]) == False):    
                    array.append(gmmProbability(rightPupilData[t], y, 'right'))
                
                if((gazeGradientData.iloc[t].dropna().empty) == False):
                    array.append(mulnor(gazeGradientData.iloc[t], y))
                
                temp = logExpSum(array)
                
                # y  -> t   -> state in this time step
                # y0 -> t-1 -> state 1 time step ago
                # x  -> t-2 -> state 2 time steps ago
                
                if (temp > maximum):
                    maximum = temp
                    state = y0
                
                
            V[t][y] = maximum
            newpath[y] = path[state] + [y]

        # Don't need to remember the old paths
        path = newpath
        
        
    #print_dptable(V)
    (prob, state) = max((V[t][y], y) for y in states)
    # return (prob, path[state])
    #print(prob, path[state])
    #print(type(path[state]))
    return( path[state] ) 



##------------------------- Saving in a .csv file ----------##
    
def exportcsv(path, A, B):

    fields = ['Prediction','A','B']

    with open ('output.csv','w') as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for i in range(len(A)):
            
            if((A[i] or B[i])== '0_unstated' ):     #Intially, after ScreenRecordStart, there were no annotations made - 
                continue                            #  for some rows, so we skip those rows
            
            else:
                a = A[i].split('_')
                b = B[i].split('_')
                         
                data = [{ 'Prediction': path[i], 'A':a[1], 'B':b[1]  }]    
                writer.writerows(data)


    file.close()


#--------------- log Exp trick ----------#

def logExpSum(arr ):
    #find maximum of array assuming the array passed is already containg log values
    maxVal =0
    maxVal= findMaxArray(arr)
    res = 0
    for i in range(0, len(arr)):
        res += m.exp (arr[i] - maxVal) 
    return (m.log(res)+ maxVal)    

    
    
def findMaxArray(arr):
    maxValue = arr[0]
    for i in range(0, len(arr)):
        if(maxValue <arr[i]):
            maxValue = arr[i]
    return maxValue
    



#-------------- Main--------#
def main():
    path = viterbi(gazeEventData, leftPupilData, rightPupilData, gazeGradientData, states, start_probability, transition, emission_probability)

    #print(len(path))
    #print(len(gd1))
    #print(len(gd2))
    exportcsv(path, gd1, gd2)
    

if __name__ == '__main__':
    main()