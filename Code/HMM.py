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



# Modeling
states = ('Scanning', 'Skimming', 'Reading', 'MediaView', 'Unknown')
start_probability = {'Scanning': -1.72276659775, 'Skimming': float("-inf"), 'Reading': -0.62415430908, 'MediaView': float("-inf"), 'Unknown':  -1.25276296851}


#Dummy data for testing GazeEventType
observations = ('Fixation','Saccade','Fixation','Fixation','Fixation','Fixation','Fixation')    #First 7 sates fixation
gr1 = ['1_Scanning','1_Scanning','1_Unknown','1_Unknown','1_Unknown','1_Reading','1_Skimming']
gr2 = ['1_Skimming','1_Reading','1_Unknown','1_MediaView','1_Unknown','1_Reading','1_Scanning']


#-------------------- 1st order Transition Matrix ----------#

transition_probability = {
   'Scanning' : {'Scanning': -0.00232095217675443, 'Skimming': -7.88855044050366, 'Reading': -7.53491040026009, 'MediaView': -9.24499183847387, 'Unknown': -6.6357874720083},
   'Skimming' : {'Scanning': -7.48995094652083, 'Skimming': -0.00236717233575057, 'Reading': -7.22484319610759, 'MediaView': -9.11740736445761, 'Unknown': -6.9405916587526},
   'Reading'  : {'Scanning': -7.68382320280519, 'Skimming': -8.02791936453706, 'Reading': -0.00201180911939325, 'MediaView': -10.1073609062169, 'Unknown': -6.74006507623042},
   'MediaView': {'Scanning': -6.9409482463379, 'Skimming': -7.02795962332753, 'Reading': -7.8164169836918, 'MediaView': -0.00371507453616538, 'Unknown': -6.53548313822974},
   'Unknown'  : {'Scanning': -7.50610673071849, 'Skimming': -8.39992460674059, 'Reading': -7.96406202144805, 'MediaView': -10.6841605610664, 'Unknown': -0.00114590030289818}
   }

##------------------------------- MODEL FOR GAZE EVENT TYPE ---------------------------------------##

emission_probability = {
   'Scanning' : {'Fixation': -0.51042419, 'Saccade': -1.17873170, 'Unclassified':  -2.38498473},
   'Skimming' : {'Fixation': -0.71985695, 'Saccade':  -0.90357292, 'Unclassified': -2.22508255},
   'Reading' : {'Fixation': -0.29756367, 'Saccade':  -1.60167980, 'Unclassified': -2.88567604},
   'MediaView' : {'Fixation': -0.36204432, 'Saccade':  -1.40492718, 'Unclassified': -2.84106350},
   'Unknown' : {'Fixation': -1.01026657, 'Saccade': -1.33310886, 'Unclassified': -0.98826541}
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


#def normalProbability(x, mean, std_dev):
#    return ( (1/(std_dev*2.507)) * m.exp((-0.5)*m.pow( (x - mean)/std_dev , 2) ) )


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
    
    
    if(side=='left'):           #side -> decides which (left or right) pupil model are we going to use. 
        n = len(leftPupilModel[key])        
        for i in range(n):
            tempProbabs.append(logPdf(x, leftPupilModel[key][i]['mean'] , leftPupilModel[key][i]['std_dev'] )+ m.log(leftPupilModel[key][i]['weight']))
        p = logExpSum(tempProbabs)
    
    elif(side=='right'):
        n = len(rightPupilModel[key])
        for i in range(n):
            tempProbabs.append(logPdf(x, rightPupilModel[key][i]['mean'] , rightPupilModel[key][i]['std_dev'] )+ m.log(rightPupilModel[key][i]['weight']))
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
  return mn.logpdf(x, mean = MultiVariateModel[key]['MeanArray'], cov = MultiVariateModel[key]['Coovariance'])

##-----------------------------------   VITERBI IMPLIMENTATION ------------------------------------------------------##


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
#def viterbi(gazeEventData, states, start_p, trans_p, emit_p):    
    V = [{}]                        #[]-> List ; {} -> Dictionary        [{}] ->List of dictionary
    path = []
    dic = {}
   
    
    # Initialize base cases (t == 0)
    for p in states:

        array = []
        array.append(start_p[p])
        
        #Logic for skipping probilities, when data is not presents        
        if(pd.isnull(gazeEventData[0]) == False):
            array.append(emit_p[p][gazeEventData[0]])
                    
        if(pd.isnull(leftPupilData[0]) == False):    
            array.append(gmmProbability(leftPupilData[0], p, 'left'))
        if(pd.isnull(rightPupilData[0]) == False):    
            array.append(gmmProbability(rightPupilData[0], p, 'right'))
                
        if((gazeGradientData.iloc[0].dropna().empty) == False):
            array.append(mulnor(gazeGradientData.iloc[0], p))
        
        
        V[0][p] = sum(array)
        dic[p] = [p]
    path.append(dic)

   


    # Run Viterbi for (t >= 1)
    dic = {}
    for t in range(1, len(gazeEventData)):
        V.append({})

        for q in states:
            maximum = float("-inf")
            state = ''
            array = []
            for p in states:
                      # y -> t   -> state in this time step  Q
                      #y0 -> t-1 -> state 1 time step ago    P
                
                
                array = [(V[t-1][p]), trans_p[p][q] ]
                
                #Logic for skipping probilities, when data is not presents
                if(pd.isnull(gazeEventData[t]) == False):
                    array.append(emit_p[q][gazeEventData[t]])
                    
                if(pd.isnull(leftPupilData[t]) == False):    
                    array.append(gmmProbability(leftPupilData[t], q, 'left'))
                if(pd.isnull(rightPupilData[t]) == False):    
                    array.append(gmmProbability(rightPupilData[t], q, 'right'))
                
                if((gazeGradientData.iloc[t].dropna().empty) == False):
                    array.append(mulnor(gazeGradientData.iloc[t], q))
                    
                
                temp = sum(array)
                
                if (temp > maximum):
                    maximum = temp
                    state = p
                        
            
            V[t][q] = maximum
            dic[q] = state
            
        path.append(dic)
        
    # print_dptable(V)
    (prob, state) = max((V[t][y], y) for y in states)
    # return (prob, path[state])
    #print(prob, path[state])
    #print(type(path[state]))
    
    out = []
    out.append(state)
    for i in range((len(V)-1),0,-1):
        key = out[-1]
        out.append(path[i][key])
    
    out.reverse()
    
    return(out)

##------------------------- Saving in a .csv file ----------##
    
def exportcsv(path, A, B):
 
    fields = ['Prediction','A','B']

    with open ('output_1st.csv','w') as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for i in range(len(A)):
            
            if( (A[i].lower() == '0_unstated') or (B[i].lower() == '0_unstated')  ):      #Intially, after ScreenRecordStart, there were no annotations made -
                continue                             #  for some rows, so we skip those rows  
            
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
    

##--------------------- MAIN() ------------------##

def main():
    path = viterbi(gazeEventData, leftPupilData, rightPupilData, gazeGradientData, states, start_probability, transition_probability, emission_probability)
#    path = viterbi(observations, states, start_probability, transition_probability, emission_probability)
        
 #   print(path)
    exportcsv(path, gd1, gd2)
    

if __name__ == '__main__':
    main()
