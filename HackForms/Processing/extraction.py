import numpy as np
import cv2
import pandas as pd
import imutils
import pytesseract
import HackForms.Processing.nlp2 as nlp2
# from spellchecker import SpellChecker
from spellchecker import SpellChecker
def radio_identification(img, df, df_final, length,data_dict,inx,count_dict):

    group = None
    x, y, h, w = 0, 0, 0, 0
    df_temp = pd.DataFrame(columns=['sum', 'no', 'group'])
    for i, row in df[df.type == 'radio'].iterrows():
        try:
            if row['group'] != 'NaN':
                curr_group = list(row['group'])
            else:
                continue
            if isinstance(curr_group, list):
                if group != curr_group[0]:
                    df_temp = df_temp.sort_values(by=['sum'])
                    try:
                        if str(group) in data_dict["radio"]:
                            pass
                        else:
                            data_dict["radio"][str(group)] = {}
                        if df_temp.iloc[0]['no'] < 0:
                            df_final.at[length, df.loc[group]['value']] = abs(df_temp.iloc[0]['no'])
                                # if str(abs(df_temp.iloc[0]['no'])) in data_dict["labels"][str(group)]:
                                #     data_dict["labels"][str('group')][str(abs(df_temp.iloc[0]['no']))]={}
                            try:
                                data_dict["radio"][str(group)][str(abs(df_temp.iloc[0]['no']))]+=1
                            except:
                                data_dict["radio"][str(group)][str(abs(df_temp.iloc[0]['no']))]=1

                        else:
                            df_final.at[length, df.loc[group]['value']]= df.loc[df_temp.iloc[0]['no']]['value']

                            df_final.at[length, df.loc[df_temp.iloc[0]['no']]['value']] = 1#df.loc[df_temp.iloc[0]['no']]['value']
                            try:
                                data_dict["radio"][str(group)][str(df.loc[df_temp.iloc[0]['no']]['value'])]+=1
                            except:
                                data_dict["radio"][str(group)][str(df.loc[df_temp.iloc[0]['no']]['value'])]=1


                    except Exception as e:
                        print("Exception in radio identification(group is list):",e)
                    df_temp = pd.DataFrame(columns=['sum', 'no', 'group'])
                    group = curr_group[0]
                    x, y, h, w = int(row['left']), int(row['top']), int(row['height']), int(row['width'])
                    arr = img[y:y + h, x:x + w]
                    df_temp = df_temp.append({'sum': np.sum(arr), 'no': curr_group[1], 'group': curr_group[0]},
                                             ignore_index=True)
                else:
                    x, y, h, w = int(row['left']), int(row['top']), int(row['height']), int(row['width'])
                    arr = img[y:y + h, x:x + w]
                    df_temp = df_temp.append({'sum': np.sum(arr), 'no': curr_group[1], 'group': curr_group[0]},
                                             ignore_index=True)
        except Exception as e:
            print("Exception in radio identification:", e)
    try:
        df_temp = df_temp.sort_values(by=['sum'])
        if str(group) in data_dict["radio"]:
            pass
        else:
            data_dict["radio"][str(group)] = {}
        if df_temp.iloc[0]['no'] < 0:
            df_final.at[length, df.loc[group]['value']] = abs(df_temp.iloc[0]['no'])
            try:
                data_dict["radio"][str(group)][str(abs(df_temp.iloc[0]['no']))] += 1
            except:
                data_dict["radio"][str(group)][str(abs(df_temp.iloc[0]['no']))] = 1
        else:
            df_final.at[length, df.loc[df_temp.iloc[0]['no']]['value']] = 1
            df_final.at[length, df.loc[group]['value']] = df.loc[df_temp.iloc[0]['no']]['value']
            try:
                data_dict["radio"][str(group)][str(df.loc[df_temp.iloc[0]['no']]['value'])] += 1
            except:
                data_dict["radio"][str(group)][str(df.loc[df_temp.iloc[0]['no']]['value'])] = 1

    except Exception as e:
        print("Exception in filled radio identification:",e)
    # print(df_final)
    print('\n===============\n',data_dict)
    if 'None' in data_dict['radio']:
        del data_dict['radio']['None']
    return df_final,data_dict


def checkbox_identification(img, df, df_final, length,data_dict):

    _, threshold = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY_INV)
    group = None
    x, y, h, w = 0, 0, 0, 0
    df_temp = pd.DataFrame(columns=['sum', 'no', 'group'])
    for i, row in df[df.type == 'checkbox'].iterrows():
        #        if abs(y-row[0][2])<diff:
        if row['group'] != 'NaN' or row['group'] != -1000:
            curr_group = list(row['group'])  # literal_eval(row['group'])
        else:
            continue
        if isinstance(curr_group, list):
            try:
                if group != curr_group[0]:
                    df_temp = df_temp.sort_values(by='sum', ascending=False)
                    try:
                        max = df_temp.iloc[0]['sum']
                        max = max / 2

                        if str(group) in data_dict["checkbox"]:
                            pass
                        else:
                            data_dict["checkbox"][str(group)] = {}

                        for _, rows in df_temp.iterrows():
                            if rows['sum'] > max:
                                df_final.at[length, df.loc[rows['no']]['value']] = 't'

                                try:
                                    data_dict["checkbox"][str(group)][str(df.loc[rows['no']]['value'])] += 1
                                except:
                                    data_dict["checkbox"][str(group)][str(df.loc[rows['no']]['value'])] = 1

                            else:
                                df_final.at[length, df.loc[rows['no']]['value']] = 'f'
                    except Exception as e:
                        print("Exception in checkbox identification(group is list)",e)
                    df_temp = pd.DataFrame(columns=['sum', 'no', 'group'])
                    group = curr_group[0]
                    x, y, h, w = int(row['left']), int(row['top']), int(row['height']), int(row['width'])
                    arr = threshold[y + int(0.25 * h):y + h - int(0.25 * h), x + int(0.25 * w):x + w - int(0.25 * w)]
                    df_temp = df_temp.append({'sum': np.sum(arr), 'no': abs(curr_group[1]), 'group': curr_group[0]},
                                             ignore_index=True)
                    # cv2.imshow("crop", threshold[y + int(0.25 * h):y + h - int(0.25 * h),
                    #                    x + int(0.25 * w):x + w - int(0.25 * w)])
                    # cv2.waitKey(0)
                    cv2.destroyAllWindows()

                else:
                    x, y, h, w = int(row['left']), int(row['top']), int(row['height']), int(row['width'])
                    arr = threshold[y + int(0.25 * h):y + h - int(0.25 * h), x + int(0.25 * w):x + w - int(0.25 * w)]
                    df_temp = df_temp.append({'sum': np.sum(arr), 'no': curr_group[1], 'group': abs(curr_group[0])},
                                             ignore_index=True)
                    # cv2.imshow("crop", threshold[y + int(0.25 * h):y + h - int(0.25 * h),
                    #                    x + int(0.25 * w):x + w - int(0.25 * w)])
                    # cv2.waitKey(0)
                    cv2.destroyAllWindows()
            except Exception as e:
                print("Exception in checkbox identification", e)
    try:
        df_temp = df_temp.sort_values(by=['sum'], ascending=False)
        max = df_temp.iloc[0]['sum']
        max = max / 2
        if str(group) in data_dict["checkbox"]:
            pass
        else:
            data_dict["checkbox"][str(group)] = {}
        for _, rows in df_temp.iterrows():
            if rows['sum'] > max:
                df_final.at[length, df.loc[rows['no']]['value']] = 't'
                try:
                    data_dict["checkbox"][str(group)][str(df.loc[rows['no']]['value'])] += 1
                except:
                    data_dict["checkbox"][str(group)][str(df.loc[rows['no']]['value'])] = 1
            else:
                df_final.at[length, df.loc[rows['no']]['value']] = 'f'
    except Exception as e:
        print("Exception in filled checkbox identification", e)
    # print(df_final)
    print('\n===============\n', data_dict)
    if 'None' in data_dict['checkbox']:
        del data_dict['checkbox']['None']
    return df_final,data_dict
def transformation(img, src_img, dst_img):
    rows, cols = img.shape[:2]
    src_points = np.float32(
        [[src_img[0][0], src_img[0][1]], [src_img[1][0], src_img[1][1]], [src_img[2][0], src_img[2][1]]])
    dst_points = np.float32(
        [[dst_img[0][0], dst_img[0][1]], [dst_img[1][0], dst_img[1][1]], [dst_img[2][0], dst_img[2][1]]])
    affine_matrix = cv2.getAffineTransform(src_points, dst_points)
    img_output = cv2.warpAffine(img, affine_matrix, (cols, rows))

    try:
        img_output = img_output[dst_img[0][1]:dst_img[2][1], dst_img[0][0]:dst_img[1][0]]
        # cv2.imshow('Input', img)
        # cv2.waitKey(0)
        # cv2.imshow('Output', img_output)
        # cv2.waitKey(0)
        cv2.destroyAllWindows()
    except Exception as e:
        print(e)
    return img_output

def perform_OCR(img, df, df_final, length):

    fieldsDf = df[df.type=='field']
    fieldsDf = fieldsDf.sort_values(by=['top','left'])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img.astype('uint8')
    # img = imutils.resize(img, width=1000)
    result = ''
    flag=0
    group=0
    group_result=''
    # temp_group_result=''
    final_result=''
    cmpd=0
    for i,row in fieldsDf.iterrows():
        t = row['top']
        l = row['left']
        h = row['height']
        w = row['width']

        # crop the photo and submit to tesseract
        # img = cv2.imread(img)
        cropped_img = img[t+3:t+h-6,l+5:l+w-5]
        cropped_img = cv2.medianBlur(cropped_img,3)
        # cv2.imshow('cropped img',cropped_img)
        # cv2.waitKey(0)

        # threshed= cv2.adaptiveThreshold(cropped_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        # cv2.imshow('thresholded img',threshed)
        # cv2.waitKey(0)
        result = pytesseract.image_to_string(cropped_img,config='--psm 6')
        print("from tess:",result)
        try:
            if row.group == 'NaN' or int(row.group)== -1000:        #TODO Set this correct
                final_result += ' ' + result
                df_final.at[length, 'Unassigned'] = final_result
                continue
            int(row.group)
            # print(int(row.group))
            if group == int(row.group):
                group_result += ' '+ result
                # df_final.at[length, df.loc[int(row.group)].value] = group_result
            else:
                temp_group_result = group_result
                group_result = result
                # group = int(row.group)
            # if flag==1:
                # if df.loc[int(row.group)].value == 'Date' or df.loc[int(row.group)].value == 'date' or df.loc[int(row.group)].value == 'Name' or df.loc[int(row.group)].value == 'name' or df.loc[int(row.group)].value == 'Email' or df.loc[int(row.group)].value == 'email' or df.loc[int(row.group)].value == 'Phone' or df.loc[int(row.group)].value == 'phone':
                # flag=-1
                if df.loc[int(row.group)].value == 'Date' or df.loc[int(row.group)].value == 'date' or df.loc[
                    int(row.group)].value == 'Name' or df.loc[int(row.group)].value == 'name' or df.loc[
                    int(row.group)].value == 'Email' or df.loc[int(row.group)].value == 'email' or df.loc[
                    int(row.group)].value == 'Phone' or df.loc[int(row.group)].value == 'phone':
                    pass
                else:
                    _,cmpd = nlp2.do_nlp(temp_group_result,cmpd)
                    print(' Result  ======',temp_group_result,'=========Semantic value ========',_)
                #tmp_str=group_result

                # temp_group_result= list(filter(bool, temp_group_result.splitlines()))
                # print(temp_group_result)
                # tmp_str=''
                # # [tmp_str+x for x in result]
                # for x in temp_group_result:
                #     tmp_str+=' '+x
                # print('tmpp string  ',tmp_str)
            # if flag==1:
                if group!=0:
                    # tmp_str = group_result
                    temp_group_result= list(filter(bool, temp_group_result.splitlines()))
                    print(temp_group_result)
                    tmp_str=''
                    # [tmp_str+x for x in result]
                    for x in temp_group_result:
                        tmp_str+=' '+x
                    print('tmpp string  ',tmp_str)
                    df_final.at[length, df.loc[group].value]=tmp_str
                group = int(row.group)
        except Exception as e:
            print('There\'s an exception in perform_ocr \n')
            print(e)
            # if result.isnumeric():
            #     df_final.at[length,'Unassigned'] = result
            #     print('result ====',df_final.at[length,'Unassigned'] )
            # else:
            #     final_result += ' ' + result
            #     df_final.at[length, 'Unassigned'] = final_result
            #     print('result ====', df_final.at[length, 'Unassigned'])

    _,cmpd = nlp2.do_nlp(group_result, cmpd)
    #
    print(' Result  ======', group_result, '=========Semantic value ========', _)

    group_result = list(filter(bool, group_result.splitlines()))
    # print(group_result)
    tmp_str = ''
    # [tmp_str+x for x in result]
    for x in group_result:
        tmp_str += ' ' + x
    print('tmpp string  ', tmp_str)
    df_final.at[length, df.loc[group].value] = tmp_str

    _,cmpd = nlp2.do_nlp(final_result, cmpd)
    print(' final Result  ======', final_result, '=========Semantic value ========', _)
    print('=========HOLA the final semantic o/p is ',cmpd)

    df_final.at[length,'Semantics'] = cmpd

    return df_final,cmpd
# TODO eliminate line in middle
# TODO grouping as well as insertion not working for k2