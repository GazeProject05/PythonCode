# -*- coding: utf-8 -*-
"""
Created on Fri May 17 17:05:22 2019

@author: tunji
"""

import pandas as pd
import numpy as np

df = pd.read_excel('3Proband3.xlsx')


#remove screenstartrec and screenstoprec
def choose_events(df):
    a = df['StudioEvent']
    b = []
    for x in a:
        if x == '1_Scanning':
            b.append(x)
        elif x == '2_Skimming':
            b.append(x)
        elif x == '3_Reading':
            b.append(x)
        elif x == '4_MediaView':
            b.append(x)
        elif x == '5_Unknown':
            b.append(x)
    #print(b)
            
    return b

gaze_types = choose_events(df)

#replace each reading type with an int
def new_Studio_Event(gaze_list):
    #df['StudioEvent_int']
    studioEvent_decode = []
    for x in gaze_list:
        #if x == '0_unstated':
            #studioEvent_decode.append(0)
        if x == '1_Scanning':
            studioEvent_decode.append(1)
        elif x == '2_Skimming':
            studioEvent_decode.append(2)
        elif x == '3_Reading':
            studioEvent_decode.append(3)
        elif x == '4_MediaView':
            studioEvent_decode.append(4)
        elif x == '5_Unknown':
            #5_unknown
            studioEvent_decode.append(5)
            
    #print(studioEvent_decode)
            
    return studioEvent_decode

studioEventsDecoded = new_Studio_Event(gaze_types)

#Setting array float decimal number
float_formatter = lambda x: "%.7f" % x
np.set_printoptions(formatter={'float_kind':float_formatter})
#count events
def count_events(studioEventsDecode):
    from collections import Counter
    e = studioEventsDecode
    
    
    n = len(set(e)) #number of states
    b = [[0 for _ in range(n)] for _ in range(n)]
    
    for (x,y), c in Counter(zip(e, e[1:])).items():
        b[x-1][y-1] = c
    
    probs = np.array([])
    for array_,pattern_ in zip(b,set(e)):
        for x in array_:
            probs = np.append(probs,x/(e.count(pattern_)))
    
    probs = probs.reshape(n,n, order='F')
    #for pattern_ in set(df['StudioEvent']):
        #print(pattern_)
    print(probs)
    
    
    #print(' '.join('{0:.6f}'.format(x) for x in probs))    
        
    return None
count_events(studioEventsDecoded)
    











