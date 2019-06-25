# -*- coding: utf-8 -*-
"""
Created on Sun May 19 01:41:55 2019

@author: tunji
"""

import pandas as pd
import numpy as np
#from collections import Counter
 
 
df = pd.read_excel('3Proband3.xlsx')

u = []
for col in df:
    if col not in('StudioEvent','StudioEvent_B','GazeEventType', 'GazeEventType_B'):
        u.append(col)
        
df = df.drop(columns=u)     

def emmission_probabilities(target_col,emmision_col):
    a = []
    for x in target_col:
        a.append(x)
    b = []
    for x in emmision_col:
        b.append(x)
    #b = np.array(b)
    c = []
    probabilities_ = []
    for x,y in zip(a,b):
         a_b = x,y
         c.append(a_b)
    set_c = []
    for y in set(a):
        for x in set(c):
            
            if y in x:
                d = c.count(x)/a.count(y)
                probabilities_.append(d)
                set_c.append(x)
                #print(x,'--','--',d)
    emmisions = pd.Series(probabilities_,index=set_c)
   
    
    return emmisions


def combined_emmission():
    a_emmission = emmission_probabilities(df['StudioEvent'],df['GazeEventType'])
    b_emmission = emmission_probabilities(df['StudioEvent_B'],df['GazeEventType_B'])
    
    a_b_emmission =  pd.concat([a_emmission,b_emmission]).groupby(level=0).sum()
    
    all_studio_events = ['0_unstated','1_Scanning','2_Skimming','3_Reading','4_MediaView','5_Unknown']
    emm_count = []
    emm_name = []
    for x in all_studio_events:
        for y in a_b_emmission.index:
            if x in y:
                emm_count.append(a_b_emmission[y])
                emm_name.append(y)
        for y in emm_name:
            a_b_emmission[y] = a_b_emmission[y]/sum(emm_count)
        emm_count = []
        emm_name = []
            
        #sum_ = sum(emm_count)
         
    
    
    return a_b_emmission

combined_emmission()


     