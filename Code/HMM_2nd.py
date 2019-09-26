import pandas as pd
import numpy as np
import math as m
from scipy.stats import multivariate_normal as mn
import csv

#Reading data file
df = pd.read_excel('16Proband16.xlsx')

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
observations = ('Fixation','Saccade','Saccade','Fixation','Fixation','Saccade','Fixation')
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


#-------------------- 2nd order Transition Matrix ----------#
#base e
transition = {
        'Scanning': {
                'Scanning' : {'Scanning': -0.00232644438500174, 'Skimming':-7.88618962102806, 'Reading': -7.53254958078448, 'MediaView': -9.24263101899826, 'Unknown': -6.63342665253269},
                'Skimming' : {'Scanning': float("-inf"), 'Skimming': 0.0, 'Reading': float("-inf"), 'MediaView': float("-inf"), 'Unknown': float("-inf")},
                'Reading' :  {'Scanning': float("-inf"), 'Skimming': float("-inf"), 'Reading': 0.0, 'MediaView': float("-inf"), 'Unknown': float("-inf")},
                'MediaView' :{'Scanning': float("-inf"), 'Skimming': float("-inf"), 'Reading': float("-inf"), 'MediaView': 0.0, 'Unknown': float("-inf")},
                'Unknown' :  {'Scanning': float("-inf"), 'Skimming': float("-inf"), 'Reading': float("-inf"), 'MediaView': float("-inf"), 'Unknown': 0.0}
                    },
        
        'Skimming': {
                'Scanning' : {'Scanning': 0.0, 'Skimming': float("-inf"), 'Reading': float("-inf"), 'MediaView': float("-inf"), 'Unknown': float("-inf")},
                'Skimming' : {'Scanning': -7.48754377378507, 'Skimming': -0.00237288416630221, 'Reading': -7.22243602337183, 'MediaView': -9.11500019172185, 'Unknown': -6.93818448601684},
                'Reading' :  {'Scanning': float("-inf"), 'Skimming': float("-inf"), 'Reading': 0.0, 'MediaView': float("-inf"), 'Unknown': float("-inf")},
                'MediaView' :{'Scanning': float("-inf"), 'Skimming': float("-inf"), 'Reading': float("-inf"), 'MediaView': 0.0, 'Unknown': float("-inf")},
                'Unknown' :  {'Scanning': float("-inf"), 'Skimming': float("-inf"), 'Reading': float("-inf"), 'MediaView': float("-inf"), 'Unknown': 0.0}
                    },
        
          'Reading': {
                'Scanning' : {'Scanning': 0.0, 'Skimming': float("-inf"), 'Reading': float("-inf"), 'MediaView': float("-inf"), 'Unknown': float("-inf")},
                'Skimming' : {'Scanning': float("-inf"), 'Skimming': 0.0, 'Reading': float("-inf"), 'MediaView': float("-inf"), 'Unknown': float("-inf")},
                'Reading' :  {'Scanning': -7.6817646949916, 'Skimming': -8.02586085672347, 'Reading': -0.00201595889106265, 'MediaView': -10.1053023984033, 'Unknown': -6.73800656841683},
                'MediaView' :{'Scanning': float("-inf"), 'Skimming': float("-inf"), 'Reading': float("-inf"), 'MediaView': 0.0, 'Unknown': float("-inf")},
                'Unknown' :  {'Scanning': float("-inf"), 'Skimming': float("-inf"), 'Reading': float("-inf"), 'MediaView': float("-inf"), 'Unknown': 0.0}
                    },
          
            'MediaView': {
                'Scanning' : {'Scanning': 0.0, 'Skimming': float("-inf"), 'Reading': float("-inf"), 'MediaView': float("-inf"), 'Unknown': float("-inf")},
                'Skimming' : {'Scanning': float("-inf"), 'Skimming': 0.0, 'Reading': float("-inf"), 'MediaView': float("-inf"), 'Unknown': float("-inf")},
                'Reading' :  {'Scanning': float("-inf"), 'Skimming': float("-inf"), 'Reading': 0.0, 'MediaView': float("-inf"), 'Unknown': float("-inf")},
                'MediaView' :{'Scanning': -6.93723317180174, 'Skimming': -7.02424454879137, 'Reading': -7.81270190915564, 'MediaView': -0.00372892779686396, 'Unknown': -6.53176806369357},
                'Unknown' :  {'Scanning': float("-inf"), 'Skimming': float("-inf"), 'Reading': float("-inf"), 'MediaView': float("-inf"), 'Unknown': 0.0}
                    },
            
              'Unknown': {
                'Scanning' : {'Scanning': 0.0, 'Skimming': float("-inf"), 'Reading': float("-inf"), 'MediaView': float("-inf"), 'Unknown': float("-inf")},
                'Skimming' : {'Scanning': float("-inf"), 'Skimming': 0.0, 'Reading': float("-inf"), 'MediaView': float("-inf"), 'Unknown': float("-inf")},
                'Reading' :  {'Scanning': float("-inf"), 'Skimming': float("-inf"), 'Reading': 0.0, 'MediaView': float("-inf"), 'Unknown': float("-inf")},
                'MediaView' :{'Scanning': float("-inf"), 'Skimming': float("-inf"), 'Reading': float("-inf"), 'MediaView': 0.0, 'Unknown': float("-inf")},
                'Unknown' :  {'Scanning': -7.50494623776424, 'Skimming': -8.39876411378634, 'Reading': -7.9629015284938, 'MediaView': -10.6830000681122, 'Unknown': -0.00114723164757713}
                    }
        
        }              

#print(transition['Reading']['Reading']['Unknown']) 
# t-2 -> Reading -> state 2 time steps ago
# t-1 -> Reading -> state 1 time step ago
# t   -> Unkown -> state in this time step    
       
        
##------------------------------- MODEL FOR GAZE EVENT TYPE ---------------------------------------##
#base e
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
    y = -m.log(m.sqrt(2*m.pi*deviation))- (u*u/(2*deviation))    #base e
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
  return mn.logpdf(x, mean = MultiVariateModel[key]['MeanArray'], cov = MultiVariateModel[key]['Coovariance'])  #base e
        
              
    
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
    
def viterbi(gazeEventData, leftPupilData, rightPupilData, gazeGradientData, states, start_p, trans_p, t1, emit_p):
#def viterbi(gazeEventData, states, start_p, trans_p, t1, emit_p):
    V = []                        #[]-> List ; {} -> Dictionary
    path = []
    
    
        
    
    # Initialize base cases (t = 0)
    dic = {}
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
        
                    
        dic[p] = logExpSum(array)
   
    V.append(dic)   
        
        
        
    # Initialize base cases (t = 1)
    dic = {}
    for p in states:
        for q in states:
                    
            array = []
            array.append(V[0][p])
            array.append(t1[p][q])
            
            if(pd.isnull(gazeEventData[1]) == False):
                array.append(emit_p[q][gazeEventData[1]])
                    
            if(pd.isnull(leftPupilData[1]) == False):    
                array.append(gmmProbability(leftPupilData[1], q, 'left'))
            if(pd.isnull(rightPupilData[1]) == False):    
                array.append(gmmProbability(rightPupilData[1], q, 'right'))
               
            if((gazeGradientData.iloc[1].dropna().empty) == False):
                array.append(mulnor(gazeGradientData.iloc[1], q))
            
            key = str(p) + ',' + str(q)
            dic[key] = logExpSum(array)
 
    V.append(dic)
    
           


    # Run Viterbi for (t >= 2)

    for t in range(2, len(gazeEventData)):
        dic = {}      #Variables with NO prefix, store delta values  
        dic2 = {}     #Variables with 2 as prefix, work with max prob path

# p -> t-2 -> state 2 time steps ago  
# q -> t-1 -> state 1 time step ago
# r -> t   -> state in this time step
        
        for q in states:
            for r in states:
                
                maximum = float("-inf")
                maximum2 = float("-inf")
                state = ''
                array = []    
                array2 = []    
                for p in states:
                    
                    key = str(p) + ',' + str(q)
                    array.append(V[t-1][key])
                    array2.append(V[t-1][key])
                    
                    
                    array.append(trans_p[p][q][r])
                    array2.append(trans_p[p][q][r])
                    
                    if(pd.isnull(gazeEventData[t]) == False):
                        array.append(emit_p[r][gazeEventData[t]])
                    
                    if(pd.isnull(leftPupilData[t]) == False):    
                        array.append(gmmProbability(leftPupilData[t], r, 'left'))
                    if(pd.isnull(rightPupilData[t]) == False):    
                        array.append(gmmProbability(rightPupilData[t], r, 'right'))
                
                    if((gazeGradientData.iloc[t].dropna().empty) == False):
                        array.append(mulnor(gazeGradientData.iloc[t], r))
                    
                    
                    temp = logExpSum(array)
                    temp2 = logExpSum(array2)
                    
                    if (temp > maximum):
                        maximum = temp
                    
                    if(temp2>maximum2):
                        state = p
                
                 
                key = str(q) + ',' + str(r)
                dic[key] = maximum 
                dic2[key] = state
        
        
        V.append(dic)
        path.append(dic2)

            
 #back track the most probable path    
 
    maxim = float("-inf")
    t = len(V)
    
    for q in states:
        for r in states:
            
            key =  str(q) + ',' + str(r)
            if(V[(t-1)][key] > maxim):
                
                maxim = V[t-1][key]
                last = r
                second_last = q
            

                    
                
    out = []
    out.append(last)
    out.append(second_last)
    
    for i in range( (len(V)-3),0,-1):        
        key = str(second_last) + ',' + str(last)    
        out.append( path[i][key] )
        
        last = second_last    
        second_last = path[i][key]


         
    out.reverse()           # output handling       
    return(out)


##------------------------- Saving in a .csv file ----------##
    
def exportcsv(path, A, B):
 
    path.insert(0,'')    
            
    fields = ['Prediction','A','B']

    with open ('output_2nd.csv','w') as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        
        for i in range(len(A)):
            
            
            if( (A[i].lower() == '0_unstated') or (B[i].lower() == '0_unstated')  ):   
                continue 
            
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
        res = res + m.exp (arr[i] - maxVal) 
    return (m.log(res)+ maxVal)    

    
    
def findMaxArray(arr):
    maxValue = arr[0]
    for i in range(0, len(arr)):
        if(maxValue <arr[i]):
            maxValue = arr[i]
    return maxValue
    



#-------------- Main--------#
def main():
    path = viterbi(gazeEventData, leftPupilData, rightPupilData, gazeGradientData, states, start_probability, transition, transition_probability, emission_probability)
 #   path = viterbi(observations, states, start_probability, transition, transition_probability, emission_probability)
    
#    print(path)
#    print(len(gd1))
#    print(len(gd2))
 
    exportcsv(path, gd1, gd2)

if __name__ == '__main__':
    main()