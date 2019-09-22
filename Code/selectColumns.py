# -*- coding: utf-8 -*-
"""
Created on Sat Aug 31 15:01:19 2019

@author: tunji
"""


import pandas as pd
import numpy as np

df=pd.read_csv('Proband_41_D3.txt',sep=',')

def strip_columns(df):
    u = []
    for cols in df:
        if cols not in('LocalTimeStamp_x','ParticipantName_x','StudioEvent_x','GazeEventType_x','GazePointX (MCSpx)_x','FixationPointY (MCSpx)_x','FixationPointX (MCSpx)_x','GazePointY (MCSpx)_x','PupilLeft_x','PupilRight_x','LocalTimeStamp_y','ParticipantName_y','StudioEvent_y','GazeEventType_y','GazePointX (MCSpx)_y','FixationPointY (MCSpx)_y','FixationPointX (MCSpx)_y','GazePointY (MCSpx)_y','PupilLeft_y','PupilRight_y'):
            u.append(cols)
    df= df.drop(columns=u)
    
        
    excel_file = pd.ExcelWriter('Proband_41_D3.xlsx')
    df.to_excel(excel_file, sheet_name='Sheet1', index=False)
    excel_file.save()
    excel_file.close()
        
    return None

strip_columns(df)
