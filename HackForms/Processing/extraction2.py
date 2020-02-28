import numpy as np
import cv2
import pandas as pd
import imutils
import pytesseract
# import nlp2

def radio_identification(img, df, df_final, length):
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
                        if df_temp.iloc[0]['no'] < 0:
                            df_final.at[length, df.loc[group]['value']] = abs(df_temp.iloc[0]['no'])
                        else:
                            df_final.at[length, df.loc[group]['value']] = df.loc[df_temp.iloc[0]['no']]['value']

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
        if df_temp.iloc[0]['no'] < 0:
            df_final.at[length, df.loc[group]['value']] = abs(df_temp.iloc[0]['no'])
        else:
            df_final.at[length, df.loc[group]['value']] = df.loc[df_temp.iloc[0]['no']]['value']
    except Exception as e:
        print("Exception in filled radio identification:",e)
    print(df_final)
    return df_final


def checkbox_identification(img, df, df_final, length):
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
                        for _, rows in df_temp.iterrows():
                            if rows['sum'] > max:
                                df_final.at[length, df.loc[rows['no']]['value']] = 't'
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
                    cv2.imshow("crop", threshold[y + int(0.25 * h):y + h - int(0.25 * h),
                                       x + int(0.25 * w):x + w - int(0.25 * w)])
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()

                else:
                    x, y, h, w = int(row['left']), int(row['top']), int(row['height']), int(row['width'])
                    arr = threshold[y + int(0.25 * h):y + h - int(0.25 * h), x + int(0.25 * w):x + w - int(0.25 * w)]
                    df_temp = df_temp.append({'sum': np.sum(arr), 'no': curr_group[1], 'group': abs(curr_group[0])},
                                             ignore_index=True)
                    cv2.imshow("crop", threshold[y + int(0.25 * h):y + h - int(0.25 * h),
                                       x + int(0.25 * w):x + w - int(0.25 * w)])
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
            except Exception as e:
                print("Exception in checkbox identification", e)
    try:
        df_temp = df_temp.sort_values(by=['sum'], ascending=False)
        max = df_temp.iloc[0]['sum']
        max = max / 2
        for _, rows in df_temp.iterrows():
            if rows['sum'] > max:
                df_final.at[length, df.loc[rows['no']]['value']] = 't'
            else:
                df_final.at[length, df.loc[rows['no']]['value']] = 'f'
    except Exception as e:
        print("Exception in filled checkbox identification", e)
    print(df_final)
    return df_final
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
        cv2.imshow('Input', img)
        cv2.waitKey(0)
        cv2.imshow('Output', img_output)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except Exception as e:
        print(e)
    return img_output

def perform_OCR(img, df, df_final, length):

    fieldsDf = df[df.type=='field']
    fieldsDf = fieldsDf.sort_values(by=['top','left'])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img.astype('uint8')
    img = imutils.resize(img, width=1000)
    final_result = ''
    group_result = ''
    for i,row in fieldsDf.iterrows():
        t = row['top']
        l = row['left']
        h = row['height']
        w = row['width']

        # crop the photo and submit to tesseract
        # img = cv2.imread(img)
        cropped_img = img[t+3:t+h-6,l+5:l+w-5]
        cropped_img = cv2.medianBlur(cropped_img,3)
        cv2.imshow('cropped img',cropped_img)
        cv2.waitKey(0)

        threshed= cv2.adaptiveThreshold(cropped_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        cv2.imshow('thresholded img',threshed)
        cv2.waitKey(0)
        result = pytesseract.image_to_string(cropped_img,config='--psm 7')
        print("from tess:",result)
        try:
            if group == int(row.group):
                group_result += ' '+ result
                df_final.at[length, df.loc[int(row.group)].value] = group_result
            else:
                group_result = ''
                group = int(row.group)

        except:
            final_result+=' '+ result
            df_final.at[length,'Unassigned'] = final_result
        print(final_result)
    return df_final