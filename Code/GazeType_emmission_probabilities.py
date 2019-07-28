# -*- coding: utf-8 -*-
"""
Created on Sun May 19 01:41:55 2019

@author: tunji
"""

import pandas as pd
import numpy as np
#from collections import Counter
 
 
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
            if cols not in('StudioEvent','StudioEvent_B','GazeEventType', 'GazeEventType_B'):
                u.append(cols)
        df= df.drop(columns=u)
        df_list_.append(df)
        u = []
            
    return df_list_
df_list = strip_columns(df_list)
del df1,df2,df3,df5,df6,df7,df8,df9,df10,df11,df12,df13,df14,df15,df16
#**********************************get Emmission Probabilities********************
def emmission_probabilities(df):

    def get_probabilities(target_col,emmision_col):
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

    a_b_emmission = None
    def combined_emmission(df):
        a_emmission = get_probabilities(df['StudioEvent'],df['GazeEventType'])
        b_emmission = get_probabilities(df['StudioEvent_B'],df['GazeEventType_B'])
        
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


emmission_probabilities(df1)


     