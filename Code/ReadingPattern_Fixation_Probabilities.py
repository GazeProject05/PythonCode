# -*- coding: utf-8 -*-
"""
Created on Sun May 19 01:41:55 2019

@author: tunji
"""

import pandas as pd
import numpy as np
#from collections import Counter
 
 
df = pd.read_excel('1Proband1.xlsx')
 

def fun1_(df):
    a = []
    for x in df['StudioEvent']:
        a.append(x)
    b = []
    for x in df['GazeEventType']:
        b.append(x)
    #b = np.array(b)
    c = []
    probabilities_ = []
    for x,y in zip(a,b):
         a_b = x,y
         c.append(a_b)
    
    for y in set(a):
        for x in set(c):
            if y in x:
                d = c.count(x)/a.count(y)
                probabilities_.append(d)
                print(x,'--','--',d)
   
    
    return None

fun1_(df) 






    
    
     