# -*- coding: utf-8 -*-
"""
Created on Sun Aug 18 14:31:24 2019

@author: tunji
"""
import pandas as pd
import numpy as np
import math

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


#elimiinate unneeded data
df_list = [df1,df2,df3,df5,df6,df7,df8,df10,df11,df12,df13,df14,df15,df16,df18]
def strip_columns(df_list):
    u = []
    df_list_ = []
    for df in df_list:
        for cols in df:
            if cols not in('StudioEvent_x','StudioEvent_y'):
                u.append(cols)
        df= df.drop(columns=u)
        df_list_.append(df)
        u = []
            
    return df_list_
df_list = strip_columns(df_list)

#delete original dataframes
#del df1,df2,df3,df5,df6,df7,df8,df10,df11,df12,df13,df14,df15
        
#count events
def take_counts(column_):
    #concatenate x and x+1 for a
    a_2_or = []
    for x,y in zip(column_,column_[1:]):
        j = x+y
        a_2_or.append(j)
    #concatenate xx and x
    #xxx = []
    xxx_count = 0
    count_array = np.array([])
    for x in set(a_2_or):
        for y in set(column_):
            for j,f,k in zip(column_,column_[1:],column_[2:]):
                if x+y == j+f+k:
                    xxx_count+=1
            count_array = np.append(count_array,xxx_count)
            #dict_count[x] = xxx_count
            xxx_count = 0
    
    count_array = count_array.reshape(len(set(a_2_or)),len(set(column_)), order='A')            
    matrix_df = pd.DataFrame(count_array,columns=set(column_), index=set(a_2_or))
    print("**************Count************")
    print(matrix_df) 
    
    return matrix_df

#Sum up counts of A and B Annotations
def call_for_counts(df):
    a_df = take_counts(df['StudioEvent_x'])
    b_df = take_counts(df['StudioEvent_y'])
    
    #combine both counts into one matrix    
    summed_count_matrix = pd.concat([a_df,b_df]).groupby(level=0).sum()
    print("**************A + B added together************")
    print(summed_count_matrix)
    
    return summed_count_matrix

#Get probabilities from summed counts
def combined_trans_matrix(df_list):
    #Get transmission matrix of each and put them in a list
    
    total_counts = [df.pipe(call_for_counts) for df in df_list]
    
    summed_count = pd.concat([total_counts[0],total_counts[1],total_counts[2],total_counts[3],total_counts[4],total_counts[5],
                   total_counts[6],total_counts[7],total_counts[8],total_counts[9],total_counts[10],
                   total_counts[11],total_counts[12]]).groupby(level=0).sum()
    print('******SUMMED COUNT****')
    print(summed_count)
    trans_array = np.array([])
    for col in summed_count:
        for x,(i,row) in zip(summed_count[col], summed_count.iterrows()):
            x = np.nan_to_num(x)
            row_sum = np.nan_to_num(row.sum())
            
            if x>0:
                trans_array = np.append(trans_array,math.log(x)-math.log(row_sum))
            else:
                trans_array = np.append(trans_array,0)
        
    trans_array = trans_array.reshape(len(summed_count.index),len(summed_count.columns), order='F')
    x_matrix = pd.DataFrame(trans_array,columns=summed_count.columns, index=summed_count.index)
    #**********************Print out Matrix***********************
    excel_file = pd.ExcelWriter('TP_2or_fold6.xlsx')
    x_matrix.to_excel(excel_file, sheet_name='Sheet1', index=False)
    excel_file.save()
    excel_file.close()
    
    excel_file1 = pd.ExcelWriter('counts_2or_fold6.xlsx')
    summed_count.to_excel(excel_file1, sheet_name='Sheet1', index=False)
    excel_file1.save()
    excel_file1.close()
    
    return x_matrix

combined_trans_matrix(df_list)


