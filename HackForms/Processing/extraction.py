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
        
        try:
            curr_group = literal_eval(row['group'])
#            print(curr_group)
            if group!=curr_group[0]:
                df_temp = df_temp.sort_values(by=['sum'])
                print(df_temp)
                try:
                    if df_temp.iloc[0]['no'] < 0:
                        df_final.at[0,df.loc[group]['value']]=abs(df_temp.iloc[0]['no'])
                    else:
                        print(df.loc[df_temp.iloc[0]['no']]['value'])
                        df_final.at[0,df.loc[group]['value']]=df.loc[df_temp.iloc[0]['no']]['value']

                except:
                    pass
                df_temp = pd.DataFrame(columns = ['sum','no','group'])
                group = curr_group[0]
                x,y,h,w = int(row['left']),int(row['top']),int(row['height']),int(row['width'])
                arr = img[y:y+h,x:x+w]
                df_temp=df_temp.append({'sum':np.sum(arr),'no':curr_group[1],'group':curr_group[0]}, ignore_index=True)
    
            else:
                print("same")
                x,y,h,w = int(row['left']),int(row['top']),int(row['height']),int(row['width'])
                arr = img[y:y+h,x:x+w]
                df_temp=df_temp.append({'sum':np.sum(arr),'no':curr_group[1],'group':curr_group[0]},ignore_index=True)
        except:
            pass
    try:
        if df_temp.iloc[0]['no'] < 0:
            df_final.at[0,df.loc[group]['value']]=abs(df_temp.iloc[0]['no'])
        else:
            print(df.loc[df_temp.iloc[0]['no']]['value'])
            df_final.at[0,df.loc[group]['value']]=df.loc[df_temp.iloc[0]['no']]['value']
    except:
        pass
    return df_final

def checkbox_identification(img,df,df_final):
    _, threshold = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY_INV)
    group = None
    x,y,h,w=0,0,0,0
    df_temp = pd.DataFrame(columns = ['sum','no','group'])
    for i,row in df[df.type=='checkbox'].iterrows():
#        if abs(y-row[0][2])<diff:
        print(row['group'])
        curr_group = literal_eval(row['group'])
        if isinstance(curr_group,list):
    #        try:
            if group!=curr_group[0]:
                df_temp = df_temp.sort_values(by='sum', ascending = False)
                try:
                    for _,rows in df_temp.itertuples():
                        if rows['sum']!=0:
    #                            print("here: ",df.loc[group]['value'])
                            df_final.at[0,df.loc[rows['no']]['value']]=1
                        else:
                            df_final.at[0,df.loc[rows['no']]['value']]=0
    
                except:
                     pass
                df_temp = pd.DataFrame(columns = ['sum','no','group'])
                group = curr_group[0]
                x,y,h,w = int(row['left']),int(row['top']),int(row['height']),int(row['width'])
                arr = threshold[y+int(0.25*h):y+h-int(0.25*h),x+int(0.25*w):x+w-int(0.25*w)]
                df_temp=df_temp.append({'sum':np.sum(arr),'no':abs(curr_group[1]),'group':curr_group[0]}, ignore_index=True)
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
    #        except Exception as e:
    #            print(e)
    df_temp = df_temp.sort_values(by=['sum'], ascending = False)
    print(df_temp)
    for _,rows in df_temp.iterrows():
        if rows['sum']!=0:
            print("here: ",df.loc[group]['value'])
            df_final.at[0,df.loc[rows['no']]['value']]=1
        else:
            df_final.at[0,df.loc[rows['no']]['value']]=0
    return df_final

df = pd.read_csv("res.csv", index_col = 0)
img = cv2.imread("Test1.jpg")
img = imutils.resize(img, width = 1000)
label = []
for _,row in df.iterrows():
    if row['type'] == 'label' and math.isnan(row['group']):
        label.append(row['value'])
dummy = [np.nan]*len(label)
print(dummy)
df_final = pd.DataFrame([dummy],columns =label)
print(df_final)
df_final = radio_identification(img,df,df_final)
print(df_final)
df_final = checkbox_identification(img,df,df_final)
print(df_final)
df_final.to_csv('final.csv')