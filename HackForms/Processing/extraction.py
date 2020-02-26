import numpy as np
import cv2
#import csv
import pandas as pd
import imutils
#import copy
import math
from ast import literal_eval

def radio_identification(img,df,df_final):
    group = None
    x,y,h,w=0,0,0,0
    df_temp =  pd.DataFrame(columns = ['sum','no','group'])
    for i,row in df[df.type=='radio'].iterrows():
        print("New Row")
        try:
            if row['group']!='NaN':
                print("Not Null")
                curr_group=list(row['group'])
            else:
                print("nan encountered")
                continue
            if isinstance(curr_group,list):
                print("List encountered")
                if group!=curr_group[0]:
                    print(df_temp)
                    df_temp = df_temp.sort_values(by=['sum'])
                    print(df_temp)
                    try:
                        if df_temp.iloc[0]['no'] < 0:
                            df_final.at[0,df.loc[group]['value']]=abs(df_temp.iloc[0]['no'])
                        else:
                            print(df.loc[df_temp.iloc[0]['no']]['value'])
                            df_final.at[0,df.loc[group]['value']]=df.loc[df_temp.iloc[0]['no']]['value']
    
                    except Exception as e:
                        print(e)
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
        except Exception as e:
            print(e)
    try:
        df_temp = df_temp.sort_values(by=['sum'])
        print(df_temp)
        if df_temp.iloc[0]['no'] < 0:
            df_final.at[0,df.loc[group]['value']]=abs(df_temp.iloc[0]['no'])
        else:
            print(df.loc[df_temp.iloc[0]['no']]['value'])
            df_final.at[0,df.loc[group]['value']]=df.loc[df_temp.iloc[0]['no']]['value']
    except Exception as e:
        print(e)
    return df_final

def checkbox_identification(img,df,df_final):
    _, threshold = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY_INV)
    group = None
    x,y,h,w=0,0,0,0
    df_temp = pd.DataFrame(columns = ['sum','no','group'])
    for i,row in df[df.type=='checkbox'].iterrows():
#        if abs(y-row[0][2])<diff:
        if row['group']!='NaN' or row['group']!=-1000:
            print("XXX:",list(row['group']))
            curr_group = list(row['group']) #literal_eval(row['group'])
        else:
            continue
        if isinstance(curr_group,list):
            try:
                if group!=curr_group[0]:
                    df_temp = df_temp.sort_values(by='sum', ascending = False)
                    print(df_temp)
                    try:
                        max = df_temp.iloc[0]['sum']
                        max = max/2
                        for _,rows in df_temp.iterrows():
                            if rows['sum']>max:
                                print("xYFVYVYGBYGBYF",max)
                                print("here: ",df.loc[group]['value'])
                                df_final.at[0,df.loc[rows['no']]['value']]=1
                            else:
                                df_final.at[0,df.loc[rows['no']]['value']]=0
                    except Exception as e:
                         print(e)
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
            except Exception as e:
                print(e)
    df_temp = df_temp.sort_values(by=['sum'], ascending = False)
    print(df_temp)
    max = df_temp.iloc[0]['sum']
    max = max / 2
    for _, rows in df_temp.iterrows():
        if rows['sum'] > max:
            print("xYFVYVYGBYGBYF", max)
            print("here: ",df.loc[group]['value'])
            df_final.at[0,df.loc[rows['no']]['value']]=1
        else:
            df_final.at[0,df.loc[rows['no']]['value']]=0
    return df_final

def transformation(img,src_img,dst_img):
    rows, cols = img.shape[:2]
    print(rows,cols)

    src_points = np.float32([[src_img[0],src_img[1]], [src_img[2], src_img[1]], [src_img[0], src_img[3]]])
    dst_points = np.float32([[dst_img[0],dst_img[1]], [dst_img[2], dst_img[1]], [dst_img[0], dst_img[3]]])
    affine_matrix = cv2.getAffineTransform(src_points, dst_points)
    img_output = cv2.warpAffine(img, affine_matrix, (cols, rows))

    cv2.imshow('Input', img)
    cv2.imshow('Output', img_output)
    cv2.waitKey()
    return img_output
