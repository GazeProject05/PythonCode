# -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 04:59:17 2019

@author: tunji
"""
import pandas as pd
import numpy as np
import math
from collections import Counter

df1 = pd.read_excel('1Proband1.xlsx')
df2 = pd.read_excel('2Proband2.xlsx')
df3 = pd.read_excel('3Proband3.xlsx')
df5 = pd.read_excel('5Proband5.xlsx')
df6 = pd.read_excel('6Proband6.xlsx')
df7 = pd.read_excel('7Proband7.xlsx')
df8 = pd.read_excel('8Proband8.xlsx')
df9 = pd.read_excel('9Proband9.xlsx')
df10 = pd.read_excel('10Proband10.xlsx')
df11 = pd.read_excel('11Proband11.xlsx')
df12 = pd.read_excel('12Proband12.xlsx')
df13 = pd.read_excel('13Proband13.xlsx')
df14 = pd.read_excel('14Proband14.xlsx')
df15 = pd.read_excel('15Proband15.xlsx')
df16 = pd.read_excel('16Proband16.xlsx')



df_list = [df1, df2, df3,df5,df6,df7,df8,df9,df10,df11,df12,df13,df14,df15,df16]
def strip_columns(df_list):
    u = []
    df_list_ = []
    for df in df_list:
        for cols in df:
            if cols not in('StudioEvent','StudioEvent_B'):
                u.append(cols)
        df= df.drop(columns=u)
        df_list_.append(df)
        u = []
            
    return df_list_
df_list = strip_columns(df_list)

#delete original dataframes
del df1,df2,df3,df5,df6,df7,df8,df9,df10,df11,df12,df13,df14,df15,df16

def probs_per_frame(df):
    a = Counter(df['StudioEvent'])
    a_array = []
    a0_array = []
    for x,y in zip(a.keys(),a.values()):
        a_array.append(x)
        a0_array.append(y)
        
    a_series = pd.Series(a0_array,a_array,dtype='float')
    
    #***************B****************    
    b = Counter(df['StudioEvent_B'])
    b_array = []
    b0_array = []
    for x,y in zip(b.keys(),b.values()):
        b_array.append(x)
        b0_array.append(y)
        
    b_series = pd.Series(b0_array,b_array,dtype='float')
    #sum all studioevent types together in one series
    a_b_series = pd.concat([a_series,b_series]).groupby(level=0).sum()
    #a_b_series.round(10)
    
    return a_b_series

def initial_probabilities(df_list):
    counts_list = [df.pipe(probs_per_frame) for df in df_list]#returns combined emmission count for each df
    
    total_counts = pd.concat([counts_list[0],counts_list[1],counts_list[2],counts_list[3],counts_list[4],counts_list[5],
               counts_list[6],counts_list[7],counts_list[8],counts_list[9],counts_list[10],
               counts_list[11],counts_list[12],counts_list[13],counts_list[14]]).groupby(level=0).sum()
    print('Count of each Studioevent Type:')
    print(total_counts)
    #Actual probabilities    
    sum_ = total_counts.sum()
    print('Total count is',sum_)
    print('Probabilities are:')
    for x in total_counts.index:
        total_counts[x] = round((total_counts[x]/sum_),10)
        
    return total_counts

initial_probabilities(df_list)
    
    
    