# -*- coding: utf-8 -*-
"""
Created on Fri May 17 17:05:22 2019

@author: tunji
"""

import pandas as pd
import numpy as np
import math

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
        

def gaze_probabilities(df):
    #transition_probabilities
    a = df['StudioEvent']#'StudioEvent'
    b = df['StudioEvent_B']#'StudioEvent_B'
    
    
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
    
    '''
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
    '''      
                    
    return summed_count_matrix#trans_array

#df_dict = {'first': df1, 'second': df2, 'third': df3}
#df_dict = {k: v.pipe(gaze_probabilities) for k, v in df_dict.items()}

def combined_trans_matrix(df_list):
    #Get transmission matrix of each and put them in a list
    #df_list = [df1, df2, df3]
    df_list = [df.pipe(gaze_probabilities) for df in df_list]
    
    summed_count = pd.concat([df_list[0],df_list[1],df_list[2],df_list[3],df_list[4],df_list[5],
                   df_list[6],df_list[7],df_list[8],df_list[9],df_list[10],
                   df_list[11],df_list[12],df_list[13],df_list[14]]).groupby(level=0).sum()
    print('******SUMMED COUNT****')
    print(summed_count)
    trans_array = np.array([])
    for col in summed_count:
        for x,(i,row) in zip(summed_count[col], summed_count.iterrows()):
            x = np.nan_to_num(x)
            row_sum = np.nan_to_num(row.sum())
            #j = math.log(x)-math.log(row_sum)
            if x>0:
                trans_array = np.append(trans_array,math.log(x)-math.log(row_sum))
            else:
                trans_array = np.append(trans_array,0)
    '''
    array_ = np.array([])
    for col in x:
        for val in x[col]:
            array_ = np.append(array_, val/len(df_list))'''
            
    
    trans_array = trans_array.reshape(len(summed_count.columns),len(summed_count.index), order='F')
    x_matrix = pd.DataFrame(trans_array,columns=summed_count.columns, index=summed_count.index)
    #**********************Print out Matrix***********************
    excel_file = pd.ExcelWriter('Transition_Probabilities.xlsx')
    x_matrix.to_excel(excel_file, sheet_name='Sheet1', index=False)
    excel_file.save()
    excel_file.close()
    
    return x_matrix

combined_trans_matrix(df_list)



