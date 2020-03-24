import pandas as pd

UNREACHABLE = -1000

def hackForm(csvfile):

    df = pd.read_csv(csvfile, encoding = "ISO-8859-1")
    df['group']='NaN'

#    img = cv2.imread('Test1.jpg')
#    img = imutils.resize(img, width=1000)
#    for row in df.itertuples():
#        cv2.rectangle(img, (row[1],row[2]),(row[1]+row[4],row[2]+row[3]),(0,0,255),2)
#    cv2.imwrite('temp.jpg',img)

    df = df.sort_values(by=['top', 'left']) #.reset_index(drop=True)
    df['group'] = df['group'].astype(object)
    # print("Original:\n ",df)


    min_field_height = df['height'].min()
    max_field_height = df['height'].max()
    avg_field_height = df['height'].mean()




    element=-1
    parent_group=None
    ind = -1
    # vis_dict={}
    # for x in df.i
    df['visited'] = 0
    while ind!=df.index[-1]:          #element < df.shape[0]-1:
        element+=1;labels=0;fields=0;checkboxes=0;radios=0
        in_strip_elements=0


        local_min_top = 0
        local_max_height = 0
        ERROR = (max_field_height - df.iloc[element].height )/2 #, 0.25*df.iloc[element].height)
        topy=df.iloc[element].top - ERROR
        bottomy=df.iloc[element].height+df.iloc[element].top + ERROR

        curr_df = df[(df.top>=topy) & (df.top+df.height<=bottomy)].copy()
        curr_df=curr_df.sort_values(by='left') # .reset_index(drop=True)

        print("\nOld ERROR: ",ERROR)

        for i in range(curr_df.shape[0]):
            local_min_top = min(local_min_top, curr_df.iloc[i].top)
            local_max_height = max(local_max_height, curr_df.iloc[i].height)

        ERROR = max(local_max_height - df.iloc[element].height,45/2) # (max_field_height - df.iloc[element].height )/2

        print("New ERROR: ",ERROR)

        topy = df.iloc[element].top - ERROR
        bottomy = df.iloc[element].height + df.iloc[element].top + ERROR

        curr_df = df[(df.top >= topy) & (df.top + df.height <= bottomy)].copy()
        curr_df = curr_df.sort_values(by='left')  # .reset_index(drop=True)

        print("topy:",topy," bottomy:",bottomy)
        print("\n curr_df::\n ",curr_df)


        # count number of each type of element in the strip
        for i in range(curr_df.shape[0]):
            if df.at[curr_df.index[i], 'visited'] == 0:
                element+=1
                df.at[curr_df.index[i], 'visited'] = 1
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
        print(element)





        # handle only one element in the strip
        if in_strip_elements==1:
            if curr_df.iloc[0].type=='label':
                parent_group = curr_df.index[0]
                # set parent_group to all elements in below strip

            if curr_df.iloc[0].type=='field':
                if parent_group is not None: # and is_valid_parent_group(df, topy, bottomy, parent_group):
                    df.at[curr_df.index[0],'group']=parent_group
                else:
                    print("parent_group is missing so assigning useless")
                    df.at[curr_df.index[0], 'group'] = UNREACHABLE

            if curr_df.iloc[0].type == 'checkbox' or curr_df.iloc[0].type == 'radio':
                print("single checkbox/radio is useless, so assigning useless")
                df.at[curr_df.index[0], 'group'] = UNREACHABLE


        #labels are not present in the strip
        elif labels == 0:

            if fields > 0:
                if checkboxes==0 and radios==0:
                    if parent_group is not None:# and is_valid_parent_group(df, topy, bottomy, parent_group):
                        assign_single_parent_to_all(df,curr_df,parent_group)
                if radios>0 or checkboxes>0:
                    print("Should not be the case")
                    assign_useless_to_all(df,curr_df)

            if checkboxes > 0:
                if fields==0 and radios==0:
                    if parent_group is not None:# and is_valid_parent_group(df, topy, bottomy, parent_group):
                        assign_single_parent_to_all(df,curr_df,parent_group)
                if fields > 0 or radios > 0:
                    print("Should not be the case")
                    assign_useless_to_all(df,curr_df)

            if radios > 0:
                if fields==0 and checkboxes==0:
                    if parent_group is not None:# and is_valid_parent_group(df, topy, bottomy, parent_group):
                        assign_single_parent_to_all_for_radio(df,curr_df,parent_group)
                if fields > 0 or checkboxes > 0:
                    print("Should not be the case")
                    assign_useless_to_all(df,curr_df)

        # labels are present in the strip
        elif labels > 0:

            if fields > 0:
                if checkboxes==0 and radios==0:
                    if curr_df.iloc[0].type=="label":
                        LFLFFF(df,curr_df) #  parent_group if is_valid_parent_group(parent_group,topy,bottomy) else None)
                    else:
                        if labels == fields:
                            FLFLFL(df,curr_df)
                        else:
                            print("assign all UNREACHABLE/USELESS")

                elif checkboxes>0 and radios==0:
                    if checkboxes+fields+1 == labels and parent_group is None:
                        parent_group=None;prevLabel=None
                        for index,row in curr_df.iterrows():
                            if row.type=="label":
                                prevLabel = index
                                if parent_group is None:
                                    parent_group = prevLabel
                                    prevLabel = index
                            elif row.type=="checkbox":
                                if prevLabel is not None: #is_valid_parent_group(df, topy, bottomy, parent_group) and
                                    df.at[index,"group"]=[parent_group,prevLabel]
                            elif row.type=="field":
                                if prevLabel is not None:
                                    df.at[index,"group"] = prevLabel
                                    prevLabel = None
                                    parent_group = None


                    elif checkboxes+fields == labels: #and is_valid_parent_group(df, topy, bottomy, parent_group):
                        pass                    # handle(not quite possible)
                    else:
                        pass                # handle missing_something

                elif radios>0 and checkboxes==0:
                    if radios+fields+1 == labels and parent_group is None:
                        parent_group = None;prevLabel = None
                        for index, row in curr_df.iterrows():
                            if row.type == "label":
                                prevLabel = index
                                if parent_group is None:
                                    parent_group = prevLabel
                                    prevLabel = index
                            elif row.type == "radio":
                                if prevLabel is not None: #is_valid_parent_group(df, topy, bottomy, parent_group) and
                                    df.at[index, "group"] = [parent_group, prevLabel]
                            elif row.type == "field":
                                if prevLabel is not None:
                                    df.at[index, "group"] = prevLabel
                                    prevLabel = None
                                    parent_group = None

                    elif radios+fields == labels: # and is_valid_parent_group(df, topy, bottomy, parent_group):
                        pass                    # handle(not quite possible)

                    elif labels < fields+radios:
                        parent_group = None;prevLabel = None
                        for index,row in curr_df.iterrows():
                            if row.type=="label":
                                i=1
                                prevLabel = index
                            elif row.type == "radio":
                                if prevLabel is not None:
                                    df.at[index, "group"] = [prevLabel, -1*i]
                                    i+=1
                            elif row.type == "field":
                                if prevLabel is not None:
                                    df.at[index, "group"] = prevLabel
                                    prevLabel = None
                                    parent_group = None

                    else:
                        pass                # handle missing_something

                else:
                    prevLabel = None
                    for index, row in curr_df.iterrows():
                        if row.type == "label":
                            prevLabel = index
                            if parent_group is None:
                                parent_group = prevLabel
                                # prevLabel = index
                        elif row.type == "radio" or row.type == "checkbox":
                            if prevLabel is not None: #is_valid_parent_group(df, topy, bottomy, parent_group) and
                                df.at[index, "group"] = [parent_group, prevLabel]
                        elif row.type == "field":
                            if prevLabel is not None:
                                df.at[index, "group"] = prevLabel
                                prevLabel = None
                                parent_group = None

            elif checkboxes > 0:
                if radios == 0:
                    if parent_group is not None: #is_valid_parent_group(df, topy, bottomy, parent_group):
                        if labels == checkboxes:
                            if curr_df.iloc[0].type=="label":
                                LCLCLC(df,curr_df,parent_group)
                            else:
                                CLCLCL(df,curr_df,parent_group)
                        else:
                            pass                # handle missing_something
                    else:
                        if labels == checkboxes+1 and curr_df.iloc[0].type=="label":
                            parent_group = curr_df.index[0]
                            # parent_group = curr_df.iloc[0].index
                            if curr_df.iloc[1].type=="label":
                                LLCLCLC(df,curr_df)
                            else:
                                LCLCLCL(df,curr_df)
                        else:
                            pass                # handle missing_something

                else:
                    if radios+fields < labels and parent_group is None:
                        prevLabel = None
                        for index, row in curr_df.iterrows():
                            if row.type == "label":
                                prevLabel = index
                                if parent_group is None:
                                    parent_group = prevLabel
                                    prevLabel = index
                            elif row.type == "radio" or row.type == "checkbox":
                                if prevLabel is not None:#is_valid_parent_group(df, topy, bottomy, parent_group) and
                                    df.at[index, "group"] = [parent_group, prevLabel]

            elif radios > 0:
                if labels == radios+1 and parent_group is None:
                    if curr_df.iloc[0].type=="label":
                        LLRLRLR(df,curr_df)
                    else:
                        LRLRLRL(df,curr_df)
                elif parent_group is not None and labels == radios:
                    if curr_df.iloc[0].type=="label":
                        LRLRLR(df, curr_df,parent_group)
                    else:
                        RLRLRL(df, curr_df,parent_group)
                elif labels < radios:
                    if curr_df.iloc[0].type=="label":
                        LRRRLRR(df,curr_df)
                    else:
                        #if is_valid_parent_group( df, topy, bottomy, parent_group):
                        i=1
                        for index,row in curr_df.iterrows():
                            if row.type == "radio":
                                df.at[index,"group"] = [parent_group, -1*i]
                                i+=1
                            elif row.type == "label":
                                parent_group = index
                                i=1


        ind = curr_df.index[-1]

    # print('new DF:\n', df)   #df[df.type!='label'])
    df.to_csv('res.csv')
    return df


def is_valid_parent_group(df, strip_top, strip_bottom, parent_group):
    if parent_group is not None:
        print("DEBUG: DF:\n",df)
        print("DEBUG: parent:",parent_group," top:",strip_top," bottom:",strip_bottom)
        if df.loc[parent_group].top > (2.2)*strip_top - strip_bottom and df.loc[parent_group].top+df.loc[parent_group].height < strip_top:
            return True
        else:
            # parent_group = None
            return False
    else:
        return False

def assign_single_parent_to_all(df,curr_df, parent_group):
    for i,row in curr_df.iterrows():
        df.at[i,"group"] = parent_group

def assign_single_parent_to_all_for_radio(df, curr_df, parent_group):
    i=1
    for index, row in curr_df.iterrows():
        df.at[index, "group"] = [ parent_group, -1*i]
        i+=1

def assign_useless_to_all(df, curr_df):
    for i,row in curr_df.iterrows():
        df.at[i, "group"] = UNREACHABLE




def LFLFFF(df,curr_df):
    prevIndexValue = curr_df.index[0]
    for index, row in curr_df.iloc[1:].iterrows():
        if row.type == 'label':
            prevIndexValue = index
        elif row.type == 'field':
            df.at[index, 'group'] = prevIndexValue

def FLFLFL(df, curr_df):
        prevIndex = curr_df.index[0]
        for index, row in curr_df.iloc[1:].iterrows():
            if row.type == 'label':
                df.at[prevIndex, 'group'] = index
            if row.type == 'field':
                prevIndex = index



def LCLCLC(df,curr_df,parent_group):
    prevIndexValue = curr_df.index[0]
    for index, row in curr_df.iloc[1:].iterrows():
        if row.type == 'label':
            prevIndexValue = index
        elif row.type == 'checkbox':
            if parent_group is not None:
                df.at[index, 'group'] = [parent_group, prevIndexValue]
            else:
                df.at[index, 'group'] = prevIndexValue

def LCLCLCL(df,curr_df):
    parent_group = curr_df.index[0]
    prevIndex = curr_df.index[1]
    for index, row in curr_df.iloc[2:].iterrows():
        if row.type == 'label':
            if parent_group is not None:
                df.at[prevIndex, 'group'] = [parent_group, index]
            else:
                df.at[prevIndex, 'group'] = index
        if row.type == 'checkbox':
            prevIndex = index

def LLCLCLC(df,curr_df):
    parent_group = curr_df.index[0]
    prevIndexValue = curr_df.index[1]
    for index, row in curr_df.iloc[2:].iterrows():
        if row.type == 'label':
            prevIndexValue = index
        elif row.type == 'checkbox':
            if parent_group is not None:
                df.at[index, 'group'] = [parent_group, prevIndexValue]
            else:
                df.at[index, 'group'] = prevIndexValue

def CLCLCL(df,curr_df,parent_group):
    prevIndex = curr_df.index[0]
    for index, row in curr_df.iloc[1:].iterrows():
        if row.type == 'label':
            if parent_group is not None:
                df.at[prevIndex, 'group'] = [parent_group, index]
            else:
                df.at[prevIndex, 'group'] = index
        if row.type == 'checkbox':
            prevIndex = index



def LLRLRLR(df,curr_df):
    parent_group = curr_df.index[0]
    prevIndexValue = curr_df.index[1]
    for index, row in curr_df.iloc[2:].iterrows():
        if row.type == 'label':
            prevIndexValue = index
        elif row.type == 'radio':
            if parent_group is not None:
                df.at[index, 'group'] = [parent_group, prevIndexValue]
            else:
                df.at[index, 'group'] = prevIndexValue

def LRLRLRL(df,curr_df):    #not used
    parent_group = curr_df.index[0]
    prevIndex = curr_df.index[1]
    for index, row in curr_df.iloc[2:].iterrows():
        if row.type == 'label':
            if parent_group is not None:
                df.at[prevIndex, 'group'] = [parent_group, index]
            else:
                df.at[prevIndex, 'group'] = index
        if row.type == 'radio':
            prevIndex = index

def LRRRLRR(df,curr_df):
    for index,row in curr_df.iterrows():
        if row.type=="label":
            i=1
            prevIndex = curr_df.index[0]
        elif row.type=="radio":
            df.at[index,"group"] = [prevIndex,-1*i]
            i+=1

def LRLRLR(df, curr_df, parent_group):
    prevIndexValue = curr_df.index[0]
    for index, row in curr_df.iloc[1:].iterrows():
        if row.type == 'label':
            prevIndexValue = index
        elif row.type == 'radio':
            if parent_group is not None:
                df.at[index, 'group'] = [parent_group, prevIndexValue]
            else:
                df.at[index, 'group'] = prevIndexValue

def RLRLRL(df, curr_df, parent_group):
    prevIndex = curr_df.index[0]
    for index, row in curr_df.iloc[1:].iterrows():
        if row.type == 'label':
            if parent_group is not None:
                df.at[prevIndex, 'group'] = [parent_group, index]
            else:
                df.at[prevIndex, 'group'] = index
        if row.type == 'radio':
            prevIndex = index