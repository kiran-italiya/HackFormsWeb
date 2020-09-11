import pandas as pd
import HackForms.MyUtils as MyUtils

UNREACHABLE = -1000

def hackForm(csvfile, height):

    df = pd.read_csv(csvfile, encoding = "ISO-8859-1")
    df['group']='NaN'

    df = df.sort_values(by=['top'])  # .reset_index(drop=True)
    df['group'] = df['group'].astype(object)


    max_field_height = df['height'].max()

    element = -1
    parent_group = None
    ind = -1

    df['visited'] = 0

    while len(df[df['visited']==0])>0:          #element < df.shape[0]-1:
        element+=1;labels=0;fields=0;checkboxes=0;radios=0
        in_strip_elements=0


        # local_min_top = 0
        # local_max_height = 0
        # local_max_bottom = 0
        # ERROR = (max_field_height - df.iloc[element].height )/2 #, 0.25*df.iloc[element].height)
        # topy=df.iloc[element].top - ERROR
        # bottomy=df.iloc[element].height+df.iloc[element].top + ERROR
        #
        # curr_df = df[(df.top>=topy) & (df.top+df.height<=bottomy)].copy()
        # curr_df=curr_df.sort_values(by='left') # .reset_index(drop=True)
        #
        # print("\nOld ERROR: ",ERROR)
        #
        # for i in range(curr_df.shape[0]):
        #     local_min_top = min(local_min_top, curr_df.iloc[i].top)
        #     local_max_height = max(local_max_height, curr_df.iloc[i].height)
        #     local_max_bottom = max(local_max_bottom, curr_df.iloc[i].top+curr_df.iloc[i].height)
        #
        # ERROR = max(local_min_top - df.iloc[element].top, height*(0.0225)) # (max_field_height - df.iloc[element].height )/2
        #
        # # ERROR1 = max(local_min_top - df.iloc[element].top,45/2)
        # # ERROR2 = max(local_max_bottom - (df.iloc[element].height+df.iloc[element].top), 45/2)
        # #
        # # ERROR = max(ERROR1,ERROR2)
        # # print("ERROR1: ", ERROR1," ERROR2:",ERROR2)
        #
        # print("New ERROR: ",ERROR)
        # print("Height:",height)
        #
        # topy = df.iloc[element].top - ERROR
        # bottomy = df.iloc[element].height + df.iloc[element].top + ERROR

        visIndex = df[df['visited']==0].index[0]
        curr_row = df.iloc[visIndex]

        curr_df = MyUtils.allInline(curr_row[1], curr_row[1]+curr_row[2], df)
        curr_df = curr_df.sort_values(by='left')  # .reset_index(drop=True)

        if len(curr_df)==0:
            df.at[visIndex, 'visited']=1
            continue

        # print("topy:",topy," bottomy:",bottomy)
        if parent_group is None:
            print("parent_group: None")
        else:
            print("parent_group: ",  parent_group)
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
                        assign_single_parent_to_all_for_radio(df,curr_df,parent_group)
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

                    if checkboxes + fields + 1 == labels and parent_group is None:
                        parent_group = None;prevLabel = None;prevIndex = None
                        if curr_df.iloc[0].type == "label":
                            prevLabel = curr_df.index[0]
                            flag = 0
                            if curr_df.iloc[1].type == "checkbox":
                                parent_group = prevLabel
                                prevIndex = curr_df.index[1]
                                for index, row in curr_df.iloc[2:].iterrows():
                                    if row.type == 'label':
                                        flag += 1
                                        if prevIndex is not None and flag != 2:
                                            df.at[prevIndex, 'group'] = [parent_group, index]
                                        prevLabel = index
                                    elif row.type == 'checkbox':
                                        flag -= 1
                                        prevIndex = index
                                    else:
                                        parent_group = None
                                        df.at[index, 'group'] = prevLabel

                            elif curr_df.iloc[1].type == "label":
                                parent_group = prevLabel
                                prevLabel = curr_df.index[1]
                                for index, row in curr_df[2:].iterrows():
                                    if row.type == 'label':
                                        prevLabel = index
                                    if row.type == 'checkbox':
                                        if parent_group is not None:
                                            df.at[index, 'group'] = [parent_group, prevLabel]
                                        else:
                                            df.at[index, 'group'] = prevLabel
                                    if row.type == "field":
                                        df.at[index, 'group'] = prevLabel

                            else:
                                df.at[index, "group"] = prevLabel
                                flag = 0
                                for index, row in curr_df[2:].iterrows():
                                    if row.type == "label":
                                        if flag == 1:
                                            parent_group = prevLabel
                                            flag = 0
                                        prevLabel = index
                                        flag += 1
                                        if prevIndex is not None:
                                            df.at[prevIndex, "group"] = [parent_group, index]
                                    if row.type == "checkbox":
                                        if flag == 2 and parent_group is not None:
                                            df.at[index, 'group'] = [parent_group, prevLabel]
                                        elif flag == 1 and prevLabel is not None and parent_group is None:
                                            parent_group = prevLabel
                                            prevIndex = index
                                        elif flag == 1 and parent_group is not None:
                                            prevIndex = index

                                    if row.type == "field":
                                        parent_group = None
                                        df.at[index, 'group'] = prevLabel


                    elif checkboxes+fields == labels: #and is_valid_parent_group(df, topy, bottomy, parent_group):
                        pass                    # handle(not quite possible)  # TODO handle this
                    else:
                        pass                # handle missing_something

                elif radios>0 and checkboxes==0:
                    if radios+fields+1 == labels and parent_group is None:
                        parent_group = None;prevLabel = None;prevIndex=None
                        if curr_df.iloc[0].type=="label":
                            prevLabel=curr_df.index[0]
                            flag=0
                            if curr_df.iloc[1].type=="radio":
                                parent_group = prevLabel
                                prevIndex=curr_df.index[1]
                                for index, row in curr_df.iloc[2:].iterrows():
                                    if row.type == 'label':
                                        flag+=1
                                        if prevIndex is not None and flag!=2:
                                            df.at[prevIndex, 'group'] = [parent_group, index]
                                        prevLabel = index
                                    elif row.type == 'radio':
                                        flag-=1
                                        prevIndex = index
                                    else:
                                        parent_group = None
                                        df.at[index, 'group'] = prevLabel

                            elif curr_df.iloc[1].type=="label":
                                parent_group=prevLabel
                                prevLabel = curr_df.index[1]
                                for index, row in curr_df[2:].iterrows():
                                    if row.type == 'label':
                                        prevLabel = index
                                    if row.type == 'radio':
                                        if parent_group is not None:
                                            df.at[index, 'group'] = [parent_group,prevLabel]
                                        else:
                                            df.at[index, 'group'] = prevLabel
                                    if row.type == "field":
                                        df.at[index, 'group'] = prevLabel

                            else:
                                df.at[index,"group"] = prevLabel
                                flag = 0
                                for index, row in curr_df[2:].iterrows():
                                    if row.type == "label":
                                        if flag==1:
                                            parent_group = prevLabel
                                            flag=0
                                        prevLabel = index
                                        flag+=1
                                        if prevIndex is not None:
                                            df.at[prevIndex,"group"] = [parent_group, index]
                                    if row.type == "radio":
                                        if flag==2 and parent_group is not None:
                                            df.at[index, 'group'] = [parent_group, prevLabel]
                                        elif flag==1 and prevLabel is not None and parent_group is None:
                                            parent_group = prevLabel
                                            prevIndex = index
                                        elif flag==1 and parent_group is not None:
                                            prevIndex = index

                                    if row.type == "field":
                                        parent_group = None
                                        df.at[index, 'group'] = prevLabel


                    elif radios+fields == labels: # and is_valid_parent_group(df, topy, bottomy, parent_group):
                        pass                    # handle(not quite possible)  # TODO handle this

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

                    else:
                        pass                # handle missing_something

                else:
                    parent_group = None;prevLabel = None;prevIndex = None
                    size = curr_df.shape[0]
                    if curr_df.iloc[0].type == "label":
                        parent_group = curr_df.index[0]

                        if curr_df.iloc[1].type == "label":

                            flag = 1;lal = 1;last_rd = None;change = 0
                            prevLabel = curr_df.index[1]

                            for index, row in curr_df.iloc[2:].iterrows():
                                if row.type == "radio":

                                    if last_rd is None:
                                        last_rd = 1

                                    if last_rd != 1:
                                        change = 1
                                    else:
                                        change = 0

                                    if flag == 1 and change == 1:
                                        lal = 0
                                        parent_group = prevLabel

                                    if prevLabel is not None and lal == 1 and change == 0:
                                        df.at[index, "group"] = [parent_group, prevLabel]
                                    elif lal == 0:
                                        prevIndex = index

                                    flag -= 1

                                elif row.type == "checkbox":

                                    if last_rd is None:
                                        last_rd = 0

                                    if last_rd != 0:
                                        change = 1
                                    else:
                                        change = 0

                                    if flag == 1 and change == 1:
                                        lal = 0
                                        parent_group = prevLabel

                                    if prevLabel is not None and lal == 1 and change == 0:
                                        df.at[index, "group"] = [parent_group, prevLabel]
                                    elif lal == 0:
                                        prevIndex = index

                                    flag -= 1

                                elif row.type == "label":
                                    flag += 1
                                    if flag == 2:
                                        parent_group = prevLabel
                                        prevLabel = index
                                        lal = 1
                                    else:
                                        if lal == 0:
                                            if prevIndex is not None and parent_group is not None:
                                                df.at[prevIndex, "group"] = [parent_group, index]
                                        else:
                                            prevLabel = index

                                elif row.type == "field":

                                    if flag == 1 and prevLabel is not None:
                                        df.at[index,"group"] = prevLabel
                                        parent_group = None
                                        last_rd = -1; lal=1

                                    flag -= 1

                        elif curr_df.iloc[1].type == "checkbox" or curr_df.iloc[1].type == "radio":

                            flag = 0;lal = 0;last_rd = None;change = 0
                            prevIndex = curr_df.index[1]

                            for index, row in curr_df.iloc[2:].iterrows():
                                if row.type == "radio":

                                    if last_rd is None:
                                        last_rd = 1

                                    if last_rd != 1:
                                        last_rd = 1
                                        change = 1
                                    else:
                                        change = 0

                                    if flag == 2 and change == 1:
                                        lal = 0
                                        parent_group = prevLabel

                                    if prevLabel is not None and lal == 1 and change == 0:
                                        df.at[index, "group"] = [parent_group, prevLabel]
                                    elif lal == 0:
                                        prevIndex = index

                                    flag -= 1

                                elif row.type == "checkbox":

                                    if last_rd is None:
                                        last_rd = 0

                                    if last_rd != 0:
                                        last_rd = 0
                                        change = 1
                                    else:
                                        change = 0

                                    if flag == 2 and change == 1:
                                        lal = 0
                                        parent_group = prevLabel

                                    if prevLabel is not None and lal == 1 and change == 0:
                                        df.at[index, "group"] = [parent_group, prevLabel]
                                    elif lal == 0:
                                        prevIndex = index

                                    flag -= 1

                                elif row.type == "label":
                                    flag += 1

                                    if flag == 2:
                                        parent_group = index
                                        flag=0;lal = 0
                                    elif flag == 3:
                                        parent_group = prevLabel
                                        lal = 1

                                    else:
                                        if lal == 0 and df.loc[prevIndex].group=="NaN":
                                            if prevIndex is not None and parent_group is not None:
                                                df.at[prevIndex, "group"] = [parent_group, index]

                                    prevLabel = index

                                elif row.type == "field":

                                    if flag == 1 and prevLabel is not None:
                                        df.at[index, "group"] = prevLabel
                                        parent_group = None
                                        last_rd = -1;lal = 1

                                    flag -= 1

                        if curr_df.iloc[1].type == "field":
                            df.at[index,"group"] = parent_group
                            parent_group = None

                            flag = 1;lal = 1;last_rd = -1;change = 0

                            for index, row in curr_df.iloc[2:].iterrows():
                                if row.type == "label":
                                    flag += 1

                                    if parent_group is None:
                                        parent_group = index

                                    if flag==2:
                                        parent_group = prevLabel
                                        lal=1
                                    else:
                                        if lal == 0 and prevIndex is not None:
                                            if prevIndex is not None and parent_group is not None:
                                                df.at[prevIndex, "group"] = [parent_group, index]
                                        elif lal == 1:
                                            prevLabel = index

                                elif row.type == "checkbox":

                                    if last_rd != 0:
                                        change = 1
                                    else:
                                        change = 0

                                    if flag == 1 and change == 1:
                                        lal = 0
                                        parent_group = prevLabel

                                    if prevLabel is not None and lal == 1 and change == 0:
                                        df.at[index, "group"] = [parent_group, prevLabel]
                                    elif lal == 0 and prevIndex is not None:
                                        prevIndex = index

                                    flag -= 1

                                elif row.type == "radio":

                                    if last_rd != 1:
                                        change = 1
                                    else:
                                        change = 0

                                    if flag == 1 and change == 1:
                                        lal = 0
                                        parent_group = prevLabel

                                    if prevLabel is not None and lal == 1 and change == 0:
                                        df.at[index, "group"] = [parent_group, prevLabel]
                                    elif lal == 0 and prevIndex is not None:
                                        prevIndex = index

                                    flag -= 1

                                elif row.type == "field":

                                    if last_rd != -1:
                                        change = 1
                                    else:
                                        change = 0

                                    if flag == 1 and change == 1:
                                        lal = 1

                                    if prevLabel is not None:
                                        df.at[index, "group"] = prevLabel
                                        parent_group = None
                                    flag -= 1


                parent_group = None    # if field present then end of the strip set parent_group None


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
                            if curr_df.iloc[1].type=="label":
                                LLCLCLC(df,curr_df)
                            else:
                                LCLCLCL(df,curr_df)
                        else:
                            pass                # handle missing_something

                else:
                    if radios+checkboxes < labels and parent_group is None:
                       parent_group = None; prevLabel=None; prevIndex= None
                       size = curr_df.shape[0]
                       if curr_df.iloc[0].type == "label":
                           parent_group=curr_df.index[0]

                           if curr_df.iloc[1].type == "label":

                               flag=1;lal=1;last_rd=None;change=0
                               prevLabel = curr_df.index[1]

                               for index,row in curr_df.iloc[2:].iterrows():
                                   if row.type == "radio":

                                       if last_rd is None:
                                           last_rd=1

                                       if last_rd==0:
                                           change=1
                                       else:
                                           change=0

                                       if flag==1 and change==1:
                                           lal=0
                                           parent_group=prevLabel


                                       if prevLabel is not None and lal==1 and change==0:
                                           df.at[index,"group"] = [parent_group,prevLabel]
                                       elif lal==0:
                                           prevIndex=index

                                       flag-=1

                                   elif row.type == "checkbox":

                                       if last_rd is None:
                                           last_rd=0

                                       if last_rd==1:
                                           change=1
                                       else:
                                           change=0

                                       if flag == 1 and change == 1:
                                           lal = 0
                                           parent_group = prevLabel

                                       if prevLabel is not None and lal==1 and change==0:
                                           df.at[index, "group"] = [parent_group, prevLabel]
                                       elif lal == 0:
                                           prevIndex = index

                                       flag-=1

                                   elif row.type == "label":
                                       flag += 1
                                       if flag == 2:
                                           parent_group = prevLabel
                                           prevLabel = index
                                           lal = 1
                                       else:
                                            if lal==0:
                                                if prevIndex is not None and parent_group is not None:
                                                    df.at[prevIndex,"group"] = [parent_group, index]
                                            else:
                                                prevLabel = index

                           elif curr_df.iloc[1].type == "checkbox" or curr_df.iloc[1].type == "radio":

                               flag = 0;lal = 0;last_rd = None;change = 0
                               prevIndex = curr_df.index[1]

                               for index, row in curr_df.iloc[2:].iterrows():
                                   if row.type == "radio":

                                       if last_rd is None:
                                           last_rd=1

                                       if last_rd == 0:
                                           change = 1
                                       else:
                                           change = 0

                                       if flag == 2 and change == 1:
                                           lal = 0
                                           parent_group = prevLabel

                                       if prevLabel is not None and lal == 1 and change == 0:
                                           df.at[index, "group"] = [parent_group, prevLabel]
                                       elif lal == 0:
                                           prevIndex = index

                                       flag -= 1

                                   elif row.type == "checkbox":

                                       if last_rd is None:
                                           last_rd=0

                                       if last_rd == 1:
                                           change = 1
                                       else:
                                           change = 0

                                       if flag == 2 and change == 1:
                                           lal = 0
                                           parent_group = prevLabel

                                       if prevLabel is not None and lal == 1 and change == 0:
                                           df.at[index, "group"] = [parent_group, prevLabel]
                                       elif lal == 0:
                                           prevIndex = index

                                       flag -= 1

                                   elif row.type == "label":
                                       flag += 1
                                       if flag == 2:
                                           parent_group = prevLabel
                                           prevLabel = index
                                           lal = 0
                                       elif flag==3:
                                           parent_group = prevLabel
                                           prevLabel=index
                                           lal = 1
                                       else:
                                           if lal==0:
                                               if prevIndex is not None and parent_group is not None:
                                                   df.at[prevIndex, "group"] = [parent_group, index]
                                               else:
                                                   prevLabel = index


                    else:
                        prevLabel = None;i=1
                        for index, row in curr_df.iterrows():
                            if row.type == "label":
                                i = 1
                                prevLabel = index
                                if parent_group is None:
                                    parent_group = prevLabel
                                    prevLabel = index
                            elif row.type == "radio":
                                if prevLabel is not None:  # is_valid_parent_group(df, topy, bottomy, parent_group) and
                                    df.at[index, "group"] = [prevLabel, (-1)*i]; i+=1

                            elif row.type == "checkbox":
                                if prevLabel is not None:  # is_valid_parent_group(df, topy, bottomy, parent_group) and
                                    df.at[index, "group"] = [parent_group, prevLabel]

                parent_group = None


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
                parent_group = None


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






















