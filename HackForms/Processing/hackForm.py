import pandas as pd
import numpy as np
import csv
import json
import math
import cv2,imutils

height = 730
width = 1000

fields = []

pd.set_option('mode.chained_assignment',None)

def generateDf(filename):
    jsonObj = json.load(open(filename, 'r'))
    newJson = jsonObj[2]['Label']['objects']
    df = pd.DataFrame(columns=['top','left','height','width','type'])
    for data in newJson:
        tmp = data['bbox']
        tmp['type']=data['title']
        tmp['value']=data['value']
        df = df.append(tmp,ignore_index=True)
    df['group']='NaN'
    return df

# df = generateDf('exported_json.json')
# df = pd.read_csv("data.csv")
# df = pd.read_csv("visit-feedback2.csv")
df = pd.read_csv("data20.csv")

img = cv2.imread('test20.jpg')
img = imutils.resize(img, width=1000)
for row in df.itertuples():
    cv2.rectangle(img, (row[1],row[2]),(row[1]+row[4],row[2]+row[3]),(0,0,255),2)
cv2.imwrite('temp.jpg',img)

# print("DFFFF:\n",df)


# structure = -1   # 0: horizontal (Fields are beside labels)  1: vertical  -1: not set
# i = 0; j=0
# def isFieldInRight(i,j):
#     if(labelsdf['left'][i]+labelsdf['width'][i] <= fieldsdf['left'][j]):
#         if(fieldsdf['top'][j]+fieldsdf['height'][j] >= labelsdf['top'][i]+labelsdf['height'][i] and fieldsdf['top'][j] <= labelsdf['top'][i]+labelsdf['height'][i]):
#             # field is found at the right side with proper place
#             return True
#     return False

# def isFieldBelow(i,j):
#     if(labelsdf['top'][i]+labelsdf['height'][i] <= fieldsdf['top'][j]):
#         if(fieldsdf['left'][j] <= labelsdf['left'][i] and fieldsdf['left'][j]+fieldsdf['width'][j] >= labelsdf['left'][i]+labelsdf['width'][i]):
#             # field is found below
#             return True
#     return False

# def isLabelInRight(i):
#     if(i<labelsdf.count-1):
#         if(labelsdf['left'][i]+labelsdf['width'][i] < labelsdf['left'][i+1]):
#             if(labelsdf['top'][i]+labelsdf['height'][i] >= labelsdf['top'][i+1] and labelsdf['top'][i+1]+labelsdf['height'][i+1] > labelsdf['top'][i]):
#                 # label is found at the right side with proper place
#                 return True
#     return False

# def isLabelBelow(i):
#     if(i<labelsdf.count-1):
#         if(labelsdf['top'][i]+labelsdf['height'][i] < labelsdf['top'][i+1]):
#             if(labelsdf['left'][i] <= labelsdf['left'][i+1]+labelsdf['width'][i+1] and labelsdf['left'][i]+labelsdf['width'][i] > labelsdf['left'][i+1]):
#                 # field is found below
#                 return True
#     return False


# def decideStructure(i,j):
#     if(isFieldInRight(i,j) and isLabelBelow(i,j)):
#         structure=0
#     if(isLabelBelow(i,j) and isFieldInRight(i,j)):
#         structure=1

# def elementsInRows(i,j):
#     labels=0;fields=0
#     while(labelsdf['left'][i]+labelsdf['width'][i]<600):






#ERROR   ( max_field_height - current_height )/2  strip_top = current_top - ERROR   strip_bottm = current_bottom + ERRROR


df = df.sort_values(by=['top']) #.reset_index(drop=True)
df['group'] = df['group'].astype(object)
# print("Original:\n ",df)


min_field_height = df['height'].min()
max_field_height = df['height'].max()
avg_field_height = df['height'].mean()


def assign_from_last(curr_df,parent_group,df):
    in_strip_elements=curr_df.shape[0]

    if in_strip_elements%2 == 1:          # yet to test

        parent_group = curr_df.index[0]

        if curr_df.iloc[0].type=='label':
            if curr_df.iloc[1].type == 'field':
                prevIndex = curr_df.index[1]
                for index, row in curr_df.iloc[2:].iterrows():
                    if row.type == 'label':
                        if parent_group is not None:
                            df.at[prevIndex, 'group'] = [parent_group, index]
                        else:
                            df.at[prevIndex, 'group'] = index
                    if row.type == 'field':
                        prevIndex = index

            if curr_df.iloc[1].type == 'label':
                prevIndexValue = curr_df.index[1]
                for index, row in curr_df.iloc[2:].iterrows():
                    if row.type == 'label':
                        prevIndexValue = index
                    if row.type == 'field' or row.type == 'checkbox' or row.type == 'radio':
                        if parent_group is not None:
                            df.at[index, 'group'] = [parent_group, prevIndexValue]
                        else:
                            df.at[index, 'group'] = prevIndexValue

            if curr_df.iloc[1].type == 'checkbox':
                prevIndex = curr_df.index[1]
                for index, row in curr_df.iloc[2:].iterrows():
                    if row.type == 'label':
                        if parent_group is not None:
                            df.at[prevIndex, 'group'] = [parent_group, index]
                        else:
                            df.at[prevIndex, 'group'] = index
                    if row.type == 'field' or row.type == 'checkbox' or row.type == 'radio':
                        prevIndex = index

            if curr_df.iloc[1].type == 'radio':
                prevIndex = curr_df.index[1]
                for index, row in curr_df.iloc[2:].iterrows():
                    if row.type == 'label':
                        if parent_group is not None:
                            df.at[prevIndex, 'group'] = [parent_group, index]
                        else:
                            df.at[prevIndex, 'group'] = index
                    if row.type == 'field' or row.type == 'checkbox' or row.type == 'radio':
                        prevIndex = index

        else:
            pass
            # this should not be the case


    else:
        if curr_df.iloc[0].type=='field':
            prevIndex=curr_df.index[0]
            for index,row in curr_df.iloc[1:].iterrows():
                if row.type=='label':
                    if parent_group is not None:
                        df.at[prevIndex,'group'] = [parent_group,index]
                    else:
                        df.at[prevIndex,'group']=index
                if row.type=='field':
                    prevIndex=index

        if curr_df.iloc[0].type=='label':
            prevIndexValue=curr_df.index[0]
            for index,row in curr_df.iloc[1:].iterrows():
                if row.type=='label':
                    prevIndexValue=index
                if row.type=='field' or row.type=='checkbox' or row.type=='radio':
                    if parent_group is not None:
                        df.at[index,'group']=[parent_group,prevIndexValue]
                    else:
                        df.at[index,'group']=prevIndexValue

        if curr_df.iloc[0].type=='checkbox':
            prevIndex=curr_df.index[0]
            for index,row in curr_df.iloc[1:].iterrows():
                if row.type=='label':
                    if parent_group is not None:
                        df.at[prevIndex,'group'] = [parent_group,index]
                    else:
                        df.at[prevIndex,'group']=index
                if row.type=='field' or row.type=='checkbox' or row.type=='radio':
                    prevIndex=index

        if curr_df.iloc[0].type=='radio':
            prevIndex=curr_df.index[0]
            for index,row in curr_df.iloc[1:].iterrows():
                if row.type=='label':
                    if parent_group is not None:
                        df.at[prevIndex,'group'] = [parent_group,index]
                    else:
                        df.at[prevIndex,'group']=index
                if row.type=='field' or row.type=='checkbox' or row.type=='radio':
                    prevIndex=index

    return curr_df


element=-1
parent_group=None
# print(df.shape[0])
UNREACHABLE= -1000

newDf=pd.DataFrame(columns=df.columns)

while(element < df.shape[0]-1):
    element+=1;labels=0;fields=0;checkboxes=0;radios=0
    in_strip_elements=0
    topy=0;bottomy=0

    local_min_top = 0
    local_max_bottom = 0
    # first_top=df.iloc[element].top
    # first_height=df.iloc[element].height

    ERROR = (max_field_height - df.iloc[element].height )/2 #, 0.25*df.iloc[element].height)
    topy=df.iloc[element].top - ERROR
    bottomy=df.iloc[element].height+df.iloc[element].top + ERROR

    curr_df = df[(df.top>=topy) & (df.top+df.height<=bottomy)].copy()
    curr_df=curr_df.sort_values(by='left') # .reset_index(drop=True)

    # print("\nOld ERROR: ",ERROR)
    # #
    # for i in range(curr_df.shape[0]):
    #     local_min_top = min(local_min_top, curr_df.iloc[i].height)
    #     local_max_bottom = max(local_max_bottom, curr_df.iloc[i].height + curr_df.iloc[i].top)
    #
    # ERROR = (local_max_bottom - local_min_top - df.iloc[element].height) / 2  # (max_field_height - df.iloc[element].height )/2
    #
    # print("New ERROR: ",ERROR)
    #
    # topy = df.iloc[element].top - ERROR
    # bottomy = df.iloc[element].height + df.iloc[element].top + ERROR
    #
    # curr_df = df[(df.top >= topy) & (df.top + df.height <= bottomy)].copy()
    # curr_df = curr_df.sort_values(by='left')  # .reset_index(drop=True)

    print("topy:",topy," bottomy:",bottomy)
    print("\n curr_df::\n ",curr_df)

    for i in range(curr_df.shape[0]):
        element+=1
        in_strip_elements+=1
        if curr_df.iloc[i].type=='label':
            labels+=1
        if curr_df.iloc[i].type=='field':
            fields+=1
        if curr_df.iloc[i].type=='checkbox':
            checkboxes+=1
        if curr_df.iloc[i].type == 'radio':
            radios += 1
    element-=1


    if in_strip_elements==1:
        if curr_df.iloc[0].type=='label':
            parent_group=curr_df.index[0]
            # set parent_group to all elements in below strip

        if curr_df.iloc[0].type=='field':
            if parent_group is not None:
                df.at[curr_df.index[0],'group']=parent_group
            else:
                print("parent_group is missing so assigning useless")
                df.at[curr_df.index[0], 'group'] = UNREACHABLE

        if curr_df.iloc[0].type == 'checkbox' or curr_df.iloc[0].type == 'radio':
            print("single checkbox/radio is useless, so assigning useless")
            df.at[curr_df.index[0], 'group'] = UNREACHABLE


    if (labels>0 and ( fields>0 or checkboxes>0)):    # or radios>0)):
        curr_df=assign_from_last(curr_df,parent_group,df)

    if(radios>0):
        if (labels>0 and labels==radios+1) or (labels==radios and parent_group is not None):    #perfect result
            curr_df=assign_from_last(curr_df,parent_group,df)

        else:
            if labels==0 and parent_group is not None:
                i = 1
                for index, row in curr_df.loc[curr_df['type'] == 'radio'].iterrows():
                    df.at[index,'group'] = [parent_group, i * -1]
                    i += 1
            elif labels==1:
                parent_group = curr_df.index[0]
                i = 1
                for index, row in curr_df.loc[curr_df['type'] == 'radio'].iterrows():
                    df.at[index, 'group'] = [parent_group, i * -1]
                    i += 1

            else:
                i = 1
                for index,row in curr_df.loc[curr_df['type'] == 'radio'].iterrows():
                    df.at[index, 'group'] = ['useless', i * -1]
                    i += 1



# print('new DF:\n', df)   #df[df.type!='label'])
df.to_csv('res.csv')

mappingDict={}
def create_dict_from_df(dfx):
    # print('Method create dict============')
    for index,row in dfx.iterrows():
        if type(row["group"])!=list:
            if math.isnan(row["group"]):pass
                # print('Nan')
            else:
                mappingDict[str(row["group"])]=row
        else:
            p=0

            tmpDict=tmpDict2={}
            for x in row["group"]:
                if(x==row["group"][-1]):
                    tmpDict2[str(x)] = row
                else:
                    tmpDict2[str(x)] = {}
                tmpDict2=tmpDict2[str(x)]

            nxt = next(iter(tmpDict))
            if nxt in mappingDict:
                mappingDict[nxt].update(tmpDict[nxt])
            else:
                mappingDict.update(tmpDict)
create_dict_from_df(df)
print(mappingDict)