# -*- coding: utf-8 -*-
"""
Created on Sun May 19 01:41:55 2019

@author: tunji
"""

import pandas as pd
import numpy as np
from collections import Counter
 
 
df = pd.read_excel('1Proband1.xlsx')
 

def fun1_(df):
    a = df['StudioEvent']
    a = np.array(a)
    b = df['GazeEventType']
    b = np.array(b)
    c = []
    for x,y in zip(a,b):
         a_b = x,'' ,y
         c.append(a_b)
       
    value_dict = Counter(c)
    #print(value_dict)
    values_=[]
    keys_ = []
    prob_values_ = []
    
    for x in value_dict.keys():
        keys_.append(x)
    for x in value_dict.values():
        prob_values_.append(x/(len(df)))
    for x in value_dict.values():
        values_.append(x)
    for x,y,z in zip(keys_,values_,prob_values_):
        print(x,' - ' ,y,' _ ',z)
         
    return None

fun1_(df) 



'''
a=[2,1,4,5,4,6]

b=['a','c','g','a','g','k']
r = []
for x,y in zip(a,b):
    g=x,'_',y
    r.append(g)
Counter(r)

x = df['StudioEvent'].array
    
     