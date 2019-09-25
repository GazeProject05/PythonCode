# -*- coding: utf-8 -*-
"""
Created on Sun May 19 01:41:55 2019

@author: tunji
"""

import pandas as pd
import numpy as np
import math
from decimal import Decimal
#from collections import Counter

df1 = pd.read_excel('Proband_21_D1.xlsx')
df2 = pd.read_excel('Proband_22_D2.xlsx')
df3 = pd.read_excel('Proband_23_D3.xlsx')
df5 = pd.read_excel('Proband_25_D2.xlsx')
df6 = pd.read_excel('Proband_26_D3.xlsx')
df7 = pd.read_excel('Proband_27_D1.xlsx')
df8 = pd.read_excel('Proband_28_D2.xlsx')
df10 = pd.read_excel('Proband_30_D1.xlsx')
df11 = pd.read_excel('Proband_31_D2.xlsx')
df12 = pd.read_excel('Proband_32_D3.xlsx')
df13 = pd.read_excel('Proband_33_D1.xlsx')
df14 = pd.read_excel('Proband_34_D2.xlsx')
df15 = pd.read_excel('Proband_35_D3.xlsx')
df16 = pd.read_excel('Proband_36_D1.xlsx')
df18 = pd.read_excel('Proband_38_D3.xlsx')
#df19 = pd.read_excel('Proband_39_D1.xlsx')
#df20 = pd.read_excel('Proband_40_D2.xlsx')
#df21 = pd.read_excel('Proband_41_D3.xlsx')

       
df_list = [df1,df2,df3,df5,df6,df7,df8,df10,df11,df12,df13,df14,df15,df16,df18]
def strip_columns(df_list):
    u = []
    df_list_ = []
    for df in df_list:
        for cols in df:
            if cols not in('StudioEvent','StudioEvent_B','GazeEventType'):
                u.append(cols)
        df= df.drop(columns=u)
        df_list_.append(df)
        u = []
            
    return df_list_
df_list = strip_columns(df_list)
#del df1,df2,df3,df5,df6,df7,df8,df10,df11,df12,df13,df14,df15
#**********************************get Emmission Probabilities********************
#def emmission_probabilities(df_list):
    #taking counts per df 
a_b_emmission = None
def get_probabilities(df):
    #a
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
    set_c = []
    for y in set(a):
        for x in set(c):
            
            if y in x:
                d = c.count(x)#/a.count(y)
                probabilities_.append(d)
                set_c.append(x)
                #print(x,'--','--',d)
    a_emmission = pd.Series(probabilities_,index=set_c)
    
    #b
    a = []
    for x in df['StudioEvent_B']:
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
    set_c = []
    for y in set(a):
        for x in set(c):
            
            if y in x:
                d = c.count(x)#/a.count(y)
                probabilities_.append(d)
                set_c.append(x)
                #print(x,'--','--',d)
    b_emmission = pd.Series(probabilities_,index=set_c)
    
    #concatenate both a and b
    a_b_emmission =  pd.concat([a_emmission,b_emmission]).groupby(level=0).sum()#added a and b counts
   
    return a_b_emmission
    #combine probabilities of all df
def combined_emmission(df_list):
    counts_list = [df.pipe(get_probabilities) for df in df_list]#returns combined emmission count for each df
    
    total_counts = pd.concat([counts_list[0],counts_list[1],counts_list[2],counts_list[3],counts_list[4],counts_list[5],
               counts_list[6],counts_list[7],counts_list[8],counts_list[9],counts_list[10],
               counts_list[11],counts_list[12]]).groupby(level=0).sum()
    print('*************COUNTS************************')
    print(total_counts)
    '''    
    array_ = np.array([])
    for col in x:
        for val in x[col]:
            array_ = np.append(array_, val/len(df_list))
    '''
            
    all_studio_events = ['0_unstated','1_Scanning','2_Skimming','3_Reading','4_MediaView','5_Unknown']
    emm_count = []
    emm_name = []
    j = 0
    for x in all_studio_events:
        for y in total_counts.index:
            if x in y:
                emm_count.append(total_counts[y])#counts
                emm_name.append(y)#keys
                #b = total_counts[y]
        for y in emm_name:
            #total_counts[y] = '{:f}'.format(math.log(total_counts[y])-math.log(sum(emm_count)))#return log of sum minus each value here
            total_counts[y] = format(math.log(total_counts[y])-math.log(sum(emm_count)),'.8f')            
        emm_count = []
        emm_name = []
    
    #total_counts.describe()

    #**********************Print out Matrix***********************
    excel_file = pd.ExcelWriter('GazeEmmission_fold6.xlsx')
    total_counts.to_excel(excel_file, sheet_name='Sheet1', index=False)
    excel_file.save()
    excel_file.close()

    return total_counts
    

combined_emmission(df_list)


