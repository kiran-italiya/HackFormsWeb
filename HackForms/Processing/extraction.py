import numpy as np
import cv2
import csv
import pandas as pd
import imutils
import copy
import math
from ast import literal_eval

def radio_identification(img,df,df_final):
    diff = 10
    group = None
    x,y,h,w=0,0,0,0
    df_temp = pd.DataFrame(columns = ['sum','no','group'])
    for i,row in df[df.type=='radio'].iterrows():
#        if abs(y-row[0][2])<diff:
        curr_group = literal_eval(row['group'])
        if group!=curr_group[0]:
            df_temp = df_temp.sort_values(by=['sum'])
            print(df_temp)
            try:
                df_final[df.loc[group]['value']][0]=df_temp.iloc[0]['no']
            except:
                pass
            df_temp = pd.DataFrame(columns = ['sum','no','group'])
            group = curr_group[0]
            x,y,h,w = int(row['left']),int(row['top']),int(row['height']),int(row['width'])
            arr = img[y:y+h,x:x+w]
            df_temp=df_temp.append({'sum':np.sum(arr),'no':curr_group[1],'group':abs(curr_group[0])}, ignore_index=True)

        else:
            print("same")
            x,y,h,w = int(row['left']),int(row['top']),int(row['height']),int(row['width'])
            arr = img[y:y+h,x:x+w]
            df_temp=df_temp.append({'sum':np.sum(arr),'no':curr_group[1],'group':abs(curr_group[0])},ignore_index=True)
    return df_final

def checkbox_identification(img,df,df_final):
    _, threshold = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY_INV)
    diff = 10
    group = None
    x,y,h,w=0,0,0,0
    df_temp = pd.DataFrame(columns = ['sum','no','group'])
    for i,row in df[df.type=='checkbox'].iterrows():
#        if abs(y-row[0][2])<diff:
        curr_group = literal_eval(row['group'])
        if group!=curr_group[0]:
            df_temp = df_temp.sort_values(by=['sum'])
            print(df_temp)
            try:
                for _,rows in df_temp.itertuples():
                    if rows['sum']!=0:
                        print("here: ",df.loc[group]['value'])
                        df_final[df.loc[group]['value']][0]=rows['no']


            except:
                pass
            df_temp = pd.DataFrame(columns = ['sum','no','group'])
            group = curr_group[0]
            x,y,h,w = int(row['left']),int(row['top']),int(row['height']),int(row['width'])
            arr = threshold[y+int(0.25*h):y+h-int(0.25*h),x+int(0.25*w):x+w-int(0.25*w)]
            df_temp=df_temp.append({'sum':np.sum(arr),'no':curr_group[1],'group':abs(curr_group[0])}, ignore_index=True)
            cv2.imshow("crop",threshold[y+int(0.25*h):y+h-int(0.25*h),x+int(0.25*w):x+w-int(0.25*w)])
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        else:
            print("same")
            x,y,h,w = int(row['left']),int(row['top']),int(row['height']),int(row['width'])
            arr = threshold[y+int(0.25*h):y+h-int(0.25*h),x+int(0.25*w):x+w-int(0.25*w)]
            df_temp=df_temp.append({'sum':np.sum(arr),'no':curr_group[1],'group':abs(curr_group[0])},ignore_index=True)
            cv2.imshow("crop",threshold[y+int(0.25*h):y+h-int(0.25*h),x+int(0.25*w):x+w-int(0.25*w)])
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    df_temp = df_temp.sort_values(by=['sum'])
    print(df_temp)
    return df_final

df = pd.read_csv("res.csv")
img = cv2.imread("test21.jpg")
img = imutils.resize(img, width = 1000)
label = []
for _,row in df.iterrows():
    print(row)
    if row['type'] == 'label' and math.isnan(row['group']):
        label.append(row['value'])
df_final = pd.DataFrame(columns =label)
print(df_final)
df_final = radio_identification(img,df,df_final)
print(df_final)
df_final = checkbox_identification(img,df,df_final)
print(df_final)