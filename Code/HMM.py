import pandas as pd
import numpy as np
import math as m

df = pd.read_csv('19Proband19.csv')
#Observation for HMM
# obs = df[['GazeEventType','PupilLeft']]

gazeData = df['GazeEventType']
b = df['PupilLeft']
pupilData = b.str.replace(',','.').astype(float)

#obs = tuple(obs)

#GroundTruth
gd1 = df['StudioEvent']
gd2 = df['StudioEvent_B']

#WHY are obs and y2 ARE THESE DIFFERENT??????
# y2 = df['GazeEventType_B']
# y3= df['GazeEventTypeDiff']


states = ('Scanning', 'Skimming', 'Reading', 'MediaView', 'Unknown')
start_probability = {'Scanning': 0.2, 'Skimming': 0.2, 'Reading': 0.2, 'MediaView': 0.2, 'Unknown': 0.2}


observations = ('Fixation','Saccade','Saccade','Fixation','Fixation','Fixation','Fixation')    #First 7 sates fixation


# A = [   [0.8, 0.05, 0.05, 0.05, 0.05],
#         [0.05, 0.8, 0.05, 0.05, 0.05],
#         [0.05, 0.05, 0.8, 0.05, 0.05],
#         [0.05, 0.05, 0.05, 0.8, 0.05],
#         [0.05, 0.05, 0.05, 0.05, 0.8]
#     ]
transition_probability = {
   'Scanning' : {'Scanning': 0.8, 'Skimming': 0.05, 'Reading': 0.05, 'MediaView': 0.05, 'Unknown': 0.05},
   'Skimming' : {'Scanning': 0.05, 'Skimming': 0.8, 'Reading': 0.05, 'MediaView': 0.05, 'Unknown': 0.05},
   'Reading' : {'Scanning': 0.05, 'Skimming': 0.05, 'Reading': 0.8, 'MediaView': 0.05, 'Unknown': 0.05},
   'MediaView' : {'Scanning': 0.05, 'Skimming': 0.05, 'Reading': 0.05, 'MediaView': 0.05, 'Unknown': 0.8},
   'Unknown' : {'Scanning': 0.05, 'Skimming': 0.05, 'Reading': 0.05, 'MediaView': 0.05, 'Unknown': 0.8}
   }


# B = [   [0.7, 0.25, 0.05],
#         [0.3, 0.1, 0.6],
#         [0.45, 0.45, 0.1],
#         [0.55, 0.35, 0.1],
#         [0.3, 0.1, 0.6]
#     ]
emission_probability = {
   'Scanning' : {'Fixation': 0.7, 'Saccade': 0.25, 'Unclassified': 0.05},
   'Skimming' : {'Fixation': 0.3, 'Saccade': 0.1, 'Unclassified': 0.6},
   'Reading' : {'Fixation': 0.45, 'Saccade': 0.45, 'Unclassified': 0.1},
   'MediaView' : {'Fixation': 0.55, 'Saccade': 0.35, 'Unclassified': 0.1},
   'Unknown' : {'Fixation': 0.3, 'Saccade': 0.1, 'Unclassified': 0.6}
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
def viterbi(gazeData, pupilData, states, start_p, trans_p, emit_p):
    V = [{}]                        #[]-> List ; {} -> Dictionary        [{}] ->List of dictionar
    path = {}

    # Initialize base cases (t == 0)
    for y in states:
        V[0][y] = start_p[y] * emit_p[y][gazeData[0]] * gmmProbability(pupilData[0] ,y)
        path[y] = [y]

    # alternative Python 2.7+ initialization syntax
    # V = [{y:(start_p[y] * emit_p[y][obs[0]]) for y in states}]
    # path = {y:[y] for y in states}

    # Run Viterbi for (t >= 1)
    for t in range(1, len(gazeData)):
        V.append({})
        newpath = {}

        for y in states:
            (prob, state) = max(  (  m.log1p(V[t-1][y0]) + m.log1p( trans_p[y0][y] ) + m.log1p( emit_p[y][gazeData[t]] ) + m.log1p(gmmProbability(pupilData[t], y0 )) , y0 ) for y0 in states   )
            V[t][y] = prob
            newpath[y] = path[state] + [y]

        # Don't need to remember the old paths
        path = newpath

    # print_dptable(V)
    (prob, state) = max((V[t][y], y) for y in states)
    # return (prob, path[state])
    print(prob, path[state])



##--------------------- MAIN() ------------------##

def main():
    viterbi(gazeData, pupilData, states, start_probability, transition_probability, emission_probability)
    # print( type(b[9]) )

if __name__ == '__main__':
    main()
