import pandas as pd
import numpy as np
import math as m
from scipy.stats import multivariate_normal as mn
import csv

df = pd.read_csv('19Proband19.csv')
#Observation for HMM
# obs = df[['GazeEventType','PupilLeft']]

gazeEventData = df['GazeEventType']
b = df['PupilLeft']
pupilData = b.str.replace(',','.').astype(float)
gazeGradientData = df[['GazePointX (MCSpx)','GazePointY (MCSpx)']]

#obs = tuple(obs)

#GroundTruth
gd1 = df['StudioEvent']
gd2 = df['StudioEvent_B']

#WHY are obs and y2 ARE THESE DIFFERENT??????
# y2 = df['GazeEventType_B']
# y3= df['GazeEventTypeDiff']


states = ('Scanning', 'Skimming', 'Reading', 'MediaView', 'Unknown')
start_probability = {'Scanning': 0.17857142857, 'Skimming': 0.0, 'Reading': 0.53571428571, 'MediaView': 0.0, 'Unknown':  0.28571428571}


observations = ('Fixation','Saccade','Saccade','Fixation','Fixation','Fixation','Fixation')    #First 7 sates fixation


# A = [   [0.8, 0.05, 0.05, 0.05, 0.05],
#         [0.05, 0.8, 0.05, 0.05, 0.05],
#         [0.05, 0.05, 0.8, 0.05, 0.05],
#         [0.05, 0.05, 0.05, 0.8, 0.05],
#         [0.05, 0.05, 0.05, 0.05, 0.8]
#     ]
transition_probability = {
   'Scanning' : {'Scanning': -0.00203497384534224, 'Skimming': -7.5923661285198, 'Reading': 0, 'MediaView': -8.66020675852115, 'Unknown': -7.29690191562596},
   'Skimming' : {'Scanning': -8.62066282958238, 'Skimming': -0.0020733332554741, 'Reading': -9.40912018994665, 'MediaView': -8.24596938014096, 'Unknown': -7.45373129693679},
   'Reading' : {'Scanning': 0.0, 'Skimming': 0.0, 'Reading': 0.0, 'MediaView': 0.0, 'Unknown': 0.0},
   'MediaView' : {'Scanning': -8.49865885041396, 'Skimming': -7.40004656174585, 'Reading': 0.0, 'MediaView': -0.00341385911724501, 'Unknown': -6.90757007664806},
   'Unknown' : {'Scanning': -8.71173390693675, 'Skimming': -7.7837471352994, 'Reading': -10.6286565191188, 'MediaView': -8.38794682984285, 'Unknown': -0.0023707550246872}
   }

##------------------------------- MODEL FOR GAZE EVENT TYPE ---------------------------------------##

# B = [   [0.7, 0.25, 0.05],
#         [0.3, 0.1, 0.6],
#         [0.45, 0.45, 0.1],
#         [0.55, 0.35, 0.1],
#         [0.3, 0.1, 0.6]
#     ]
emission_probability = {
   'Scanning' : {'Fixation': 0.63248022558, 'Saccade': 0.28533467154, 'Unclassified':  0.08218510287},
   'Skimming' : {'Fixation': 0.54331139442, 'Saccade':  0.36026790895, 'Unclassified': 0.09642069662},
   'Reading' : {'Fixation': 0.76076073655, 'Saccade':  0.18806560675, 'Unclassified': 0.05117365669},
   'MediaView' : {'Fixation': 0.69896303332, 'Saccade':  0.2433309586, 'Unclassified': 0.05770600807},
   'Unknown' : {'Fixation': 0.42320040043, 'Saccade': 0.19158475227, 'Unclassified': 0.38521484729}
}



##--------------------    MODEL FOR PUPIL DATA -- AND -- FUNCTIONS TO CALCULATE PROBABILITIES  ------------------##

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


def normalProbability(x, mean, std_dev):
    return ( (1/(std_dev*2.507)) * m.exp((-0.5)*m.pow( (x - mean)/std_dev , 2) ) )


def gmmProbability(x, key):
    n = len(PupilModel[key])
    p = 0

    if(np.isnan(x)):              #checking for 0 (NAN) values
        return 0

    else:
        for i in range(n):
            p = p + ( normalProbability(x, PupilModel[key][i]['mean'] , PupilModel[key][i]['std_dev'] )  * PupilModel[key][i]['weight']  )

        return p



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
    if(np.isnan(x[0] or x[1]) ):
        return 0
    else:
        return mn.pdf(x, mean = MultiVariateModel[key]['MeanArray'], cov = MultiVariateModel[key]['Coovariance'])

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
def viterbi(gazeEventData, pupilData, gazeGradientData, states, start_p, trans_p, emit_p):
    V = [{}]                        #[]-> List ; {} -> Dictionary        [{}] ->List of dictionar
    path = {}

    # Initialize base cases (t == 0)
    for y in states:
        #USE LOG HERE
        V[0][y] = m.log1p( start_p[y])  + m.log1p(emit_p[y][gazeEventData[0]] ) + m.log1p( gmmProbability(pupilData[0] ,y) ) + m.log1p( mulnor(gazeGradientData.iloc[0], y) )
        path[y] = [y]

    # alternative Python 2.7+ initialization syntax
    # V = [{y:(start_p[y] * emit_p[y][obs[0]]) for y in states}]
    # path = {y:[y] for y in states}

    # Run Viterbi for (t >= 1)
    for t in range(1, len(gazeEventData)):
        V.append({})
        newpath = {}

        for y in states:
            (prob, state) = max(  (  m.log1p(V[t-1][y0]) + ( trans_p[y0][y] ) + m.log1p( emit_p[y][gazeEventData[t]] ) + m.log1p(gmmProbability(pupilData[t], y0))  + m.log1p(mulnor(gazeGradientData.iloc[t], y0))  , y0 ) for y0 in states   )
            #(prob, state) = max(  (  m.log1p(V[t-1][y0]) + ( trans_p[y0][y] ) + m.log1p( emit_p[y][gazeEventData[t]] ) , y0 ) for y0 in states   )
            V[t][y] = prob
            newpath[y] = path[state] + [y]

        # Don't need to remember the old paths
        path = newpath

    # print_dptable(V)
    (prob, state) = max((V[t][y], y) for y in states)
    # return (prob, path[state])
    #print(prob, path[state])
    #print(type(path[state]))
    return path[state]


##------------------------- Saving in a .csv file ----------##
    
def exportcsv(path, A, B):
    #print(len(path))
    #print(len(A))
    #print(len(B))
    fields = ['Prediction','A','B']

    with open ('output.csv','w') as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for i in range(len(A)):
            
            if((A[i] or B[i])== '0_unstated' ):
                continue
            
            else:
                a = A[i].split('_')
                b = B[i].split('_')
                         
                data = [{ 'Prediction': path[i], 'A':a[1], 'B':b[1]  }]    
                writer.writerows(data)


    file.close()
    

##--------------------- MAIN() ------------------##

def main():
    path = viterbi(gazeEventData, pupilData, gazeGradientData, states, start_probability, transition_probability, emission_probability)
    # v = mulnor(gazeGradientData.iloc[0],'Reading')
    # print( v )
    exportcsv(path, gd1, gd2)
    

if __name__ == '__main__':
    main()
