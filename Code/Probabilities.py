# -*- coding: utf-8 -*-
"""
Created on Fri May 17 17:05:22 2019

@author: tunji
"""

import pandas as pd
import numpy as np

df = pd.read_excel('3Proband3.xlsx')

def gaze_probabilities(df):
    #transition_probabilities
    a = df['StudioEvent']
    b = df['StudioEvent_B']
    
    
    #represent each reading type in a by an int
    a_event_idx = 0
    a_unique_events = []    
    for x in set(a):
        a_unique_events.append(a_event_idx)
        a_event_idx+=1
    
    #represent each reading type in b by an int
    b_event_idx = 0
    b_unique_events = []   
    for x in set(b):
        b_unique_events.append(b_event_idx)
        b_event_idx+=1
    
    #create full int code array of entire columns
    a_decoded = []    
    for i in a:
        for event, event_code in zip(set(a),a_unique_events):
            if event == i:
                a_decoded.append(event_code)
                
    b_decoded = []
    for i in b:
        for event, event_code in zip(set(b),b_unique_events):
            if event == i:
                b_decoded.append(event_code)
        
    #Setting array float decimal number
    float_formatter = lambda x: "%.7f" % x
    np.set_printoptions(formatter={'float_kind':float_formatter})
    
    from collections import Counter
    
    #count events a
    e = a_decoded
    a_n = len(set(e)) #number of states
    a_counts = [[0 for _ in range(a_n)] for _ in range(a_n)]
    for (x,y), c in Counter(zip(e, e[1:])).items():
        a_counts[x-1][y-1] = c
            
    '''probs = np.array([])
    for array_ in b:
        for x in array_:
            probs = np.append(probs,x/sum(array_))'''
    
    #count events b
    e = b_decoded
    b_n = len(set(e)) #number of states
    b_counts = [[0 for _ in range(b_n)] for _ in range(b_n)]
    for (x,y), c in Counter(zip(e, e[1:])).items():
        b_counts[x-1][y-1] = c
    
    #put arrays in dataframes
    a_probs = np.array([])
    for array_ in a_counts:
        for x in array_:
            a_probs = np.append(a_probs,x)
    
    
    a_probs = a_probs.reshape(a_n,a_n, order='F')            
    a_matrix = pd.DataFrame(a_probs,columns=set(a), index=set(a))
    print("**************Count of Anotation A************")
    print(a_matrix)      
    
    b_probs = np.array([])
    for array_ in b_counts:
        for x in array_:
            b_probs = np.append(b_probs,x)
    
    b_probs = b_probs.reshape(b_n,b_n, order='F')            
    b_matrix = pd.DataFrame(b_probs,columns=set(b), index=set(b))
    print("**************Count of Anotation B************")
    print(b_matrix) 
    
  
    #combine both counts into one matrix    
    summed_count_matrix = pd.concat([a_matrix,b_matrix]).groupby(level=0).sum()
    print("**************A + B added together************")
    print(summed_count_matrix)
    
    #divide along indexes
    trans_array = np.array([])
    for col in summed_count_matrix:
        for x,(i,row) in zip(summed_count_matrix[col], summed_count_matrix.iterrows()):
            #lambda i,row: for i,row in summed_count_matrix.iterrows():
            x = np.nan_to_num(x)
            row_sum = np.nan_to_num(row.sum())
            trans_array = np.append(trans_array,x/row_sum)
    trans_array = trans_array.reshape(len(summed_count_matrix.columns),len(summed_count_matrix.index), order='F')            
    trans_matrix = pd.DataFrame(trans_array,columns=summed_count_matrix.columns, index=summed_count_matrix.index)
    print("**************Actual probabilities************")
    print(trans_matrix) 
            
        
    #gaze emmission probability here
    
            
    return None#trans_array

gaze_probabilities(df)






