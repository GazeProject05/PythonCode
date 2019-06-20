# -*- coding: utf-8 -*-
"""
Created on Fri May 17 17:05:22 2019

@author: tunji
"""

import pandas as pd
import numpy as np

df = pd.read_excel('1Proband1.xlsx')

def gaze_probabilities(df):
    #transition_probabilities
    a = df['StudioEvent']
    b = df['StudioEvent_B']
    gaze_list = []
    for x ,y in zip(a,b):
        combined = x,y
        gaze_list.append(combined)
    

    #replace each combination with an int
    event_idx = 0
    studioEvent_decode = []
    decoded = []
    for x in set(gaze_list):
        studioEvent_decode.append(event_idx)
        event_idx+=1
    for i in gaze_list:
        for event, event_code in zip(set(gaze_list),studioEvent_decode):
            if event == i:
                decoded.append(event_code)
        
    #Setting array float decimal number
    float_formatter = lambda x: "%.7f" % x
    np.set_printoptions(formatter={'float_kind':float_formatter})
    #count events
    from collections import Counter
    e = decoded
    n = len(set(e)) #number of states
    b = [[0 for _ in range(n)] for _ in range(n)]
    for (x,y), c in Counter(zip(e, e[1:])).items():
        b[x-1][y-1] = c
            
    probs = np.array([])
    for array_ in b:
        for x in array_:
            probs = np.append(probs,x/sum(array_))
                
    probs = probs.reshape(n,n, order='F')
        
    #Print out matrix
    matrix_ = pd.DataFrame(probs,columns=set(gaze_list), index=set(gaze_list))
    print(matrix_)
    
    #prior probabilities here
    prior_list = []
    count1 = Counter(gaze_list)
    for x in count1.values():
        prior_list.append(x)
    prior_probabilities = pd.Series(prior_list,index=set(gaze_list))
    print(prior_probabilities)
    
    #gaze emmission probability here
            
            
    return None#probs, prior_list

gaze_probabilities(df)


    











