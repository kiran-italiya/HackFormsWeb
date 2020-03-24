import cv2
import numpy as np
import pandas as pd
import HackForms.Processing.extraction as extraction
import copy
import pytesseract

def contour(image):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, threshold = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    return contours


def findCircle(cnts, img1):
    if len(cnts) == 0:
        print("No contour found!!")
    else:
        img = img1.copy()
        circles = []
        # it to compute the minimum enclosing circle andSub1v4V
        for c in cnts:
            ((x, y), radius) = cv2.minEnclosingCircle(c)
            c_area = cv2.contourArea(c)
            c_area = c_area / ((3.14) * radius * radius)
            if radius > 10:
                if radius < 50:
                    if c_area > 0.8:
                        circles.append(
                            [int(x) - int(radius), int(y) - int(radius), int(2 * radius), int(2 * radius), "radio",
                             np.nan, 0])
                        cv2.circle(img, (int(x), int(y)), int(radius), (0, 0, 255), 1)
        #cv2.imshow("contour", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        cv2.imwrite("circle.png", img)
        return circles
def eliminate_duplicate_circle(circles,diff,img):
    df_circle = pd.DataFrame(circles, columns=['X1', 'Y1', 'R1', 'R2',"type", "value", "group"])
    df_circle = df_circle.sort_values(by=['Y1', 'X1']).reset_index(drop=True)
    # print(df_circle)
    index_curr, x, y, r = 0, 0, 0, 0
    for index_r,row in df_circle.iterrows():
        index_v = index_r
        # print(row['X1'], index_r)
        if abs(x-row['X1'])<diff and abs(y- row['Y1'])<diff:
            df_circle = df_circle.drop(index_r)
        while index_v in df_circle.index and abs(y - df_circle.loc[index_v][1]) < 10:
            if abs(x - df_circle.loc[index_v][0]) < diff and abs(y - df_circle.loc[index_v][1]) < diff:
                if x - df_circle.loc[index_v][0] > 0:
                    df_circle = df_circle.drop(index_v)
                else:
                    df_circle = df_circle.drop(index_curr)
            else:
                pass
            index_v += 1
        try:
            index_curr = index_r
            x = df_circle.loc[index_r][0]
            y = df_circle.loc[index_r][1]
            r = df_circle.loc[index_r][2]
        except:
            print("Exception occured")
    df_circle = df_circle.sort_values(by=['Y1', 'X1']).reset_index(drop=True)
    return df_circle


def linesp(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    canny = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(canny, 1, np.pi / 180, 80, None, 50, 1)
    return lines


def detect_rectangles(image):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # _, threshold = cv2.threshold(img, 215, 255, cv2.THRESH_BINARY)
    threshold = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2)
    contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    coordinate = []
    image1 = image.copy()
    cv2.drawContours(image1, contours,-1,(0,0,255),2)
    # cv2.imshow("contours", image1)
    # cv2.waitKey(0)
    cv2.destroyAllWindows()

    for cnt in contours:
        c_area = cv2.contourArea(cnt)
        x,y,w,h = cv2.boundingRect(cnt)
        area = w*h
        if c_area/area>0.9:
            coordinate.append(([x,y], [x+w,y+h]))
        # approx = cv2.approxPolyDP(cnt, 0.01 * cv2.arcLength(cnt, True), True)
        # if len(approx) == 4 and cv2.contourArea(cnt) > 50:
        #     # if abs(approx[0][0][0]-approx[1][0][0])<10 or abs(approx[0][0][0]-approx[3][0][0])<10:
        #     coordinate.append((approx[0][0], approx[2][0]))
        #			'''if 2000+co > cv2.contourArea(cnt) > co:
        #				continue
        #				co=cv2.contourArea(cnt)'''# try changing the value in place of 2000 to get outer rectangles

        for i in range(len(coordinate)):
            cv2.rectangle(img, tuple(coordinate[i][0]), tuple(coordinate[i][1]), (0, 0, 255), 1)
            cv2.circle(img, tuple(coordinate[i][0]), 5, (0, 255, 0), 5)
            cv2.circle(img, tuple(coordinate[i][1]), 5, (0, 255, 0), 5)
        # cv2.imshow("rectangle", img)
    return img, coordinate


def eliminate_duplicate_box(rec_coordinate, diff,img):

    coordinates = []
    for x, y in rec_coordinate:
        coordinates.append([x[0], x[1], y[0], y[1]])

    df_box = pd.DataFrame(coordinates, columns=['X1', 'Y1', 'X2', 'Y2'])
    df_box = df_box.sort_values(by=['Y1', 'X1']).reset_index(drop=True)
    index_curr, x1, y1, x2, y2 = 0, 0, 0, 0, 0
    for i,row in df_box.iterrows():


        if row['X1']<row['X2']:
            if row['Y1']<row['Y2']:
                pass
            elif row['Y1']>row['Y2']:
                temp = copy.deepcopy(row['Y1'])
                df_box.at[i,'Y1'] = copy.deepcopy(row['Y2'])
                df_box.at[i, 'Y2'] = temp
            else:
                print("unforeseen Condition")
        elif row['X1']>row['X2']:
            if row['Y1']<row['Y2']:
                temp = copy.deepcopy(row['X2'])
                row['X2'] = copy.deepcopy(row['X1'])
                df_box.at[i,'X2']= copy.deepcopy(row['X1'])

                df_box.at[i, 'X1'] = temp
            elif row['Y1']>row['Y2']:
                temp = copy.deepcopy(row['X2'])
                # row['X2'] = row['X1']
                # row['X1'] = temp
                df_box.at[i, 'X2'] = copy.deepcopy(row['X1'])
                df_box.at[i, 'X2'] = temp
                temp = copy.deepcopy(row['Y2'])
                # row['Y2'] = row['Y1']
                df_box.at[i, 'Y2'] = copy.deepcopy(row['Y1'])
                df_box.at[i, 'X2'] = temp
            else:
                print("unforeseen Condition")
    for i, row in df_box.iterrows():
        index_v = i
        if row['X1'] < 10 and row['Y1'] < 10:         # or (row['X1']>w and row['Y1']<10)
            df_box = df_box.drop(i)
        try:
            while abs(y1 - df_box.loc[index_v][1]) < 10:
                if index_v in df_box.index:
                    if abs(x1 - df_box.loc[index_v][0]) < diff and abs(y1 - df_box.loc[index_v][1]) < diff and abs(
                                    x2 - df_box.loc[index_v][2]) < diff and abs(y2 - df_box.loc[index_v][3]) < diff:
                        if x1 - df_box.loc[index_v][0] > 0:
                            df_box = df_box.drop(index_v)
                        else:
                            df_box = df_box.drop(index_curr)
                    else:
                        pass
                    index_v += 1
        except Exception as e:
            print(e)
            print("Something wrong")
        try:
            index_curr = i
            x1 = copy.deepcopy(df_box.loc[i][0])
            y1 = copy.deepcopy(df_box.loc[i][1])
            x2 = copy.deepcopy(df_box.loc[i][2])
            y2 = copy.deepcopy(df_box.loc[i][3])
        except:
            pass
    df_box = df_box.sort_values(by=['Y1', 'X1']).reset_index(drop=True)
    return df_box


def line_processing(line, diff, height, img1):
    horiz_lines = []
    if line is not None:
        for j in range(0, len(line)):
            l = line[j][0]
            if abs(l[1] - l[3]) < diff and l[1]>height and l[2]>20 and l[3]>height and l[0]>20:
                horiz_lines.append(line[j][0])

    df = pd.DataFrame(horiz_lines, columns=['X1', 'Y1', 'X2', 'Y2'])
    df = df.sort_values(by=['Y1', 'X1']).reset_index(drop=True)
    x1, y1, x2, y2 = 0, 0, 0, 0
    field_box = []
    index_curr = 0
    h,w = img1.shape[:2]
    img = img1.copy()

    #    """DUPLICATE LINES ELIMINATION"""
    for index_r,row in df.iterrows():
        index_v = index_r
        cv2.line(img, (row['X1'],row['Y1']),(row['X2'], row['Y2']), (0,255,0),3)
        # print('Before line\n',row)
        if abs(h - row['Y1']) < 10:
            df = df.drop(index_r)
        else:
            pass
        if row['X1']<row['X2']:
            pass
        elif row['X1']>row['X2']:
            temp = copy.deepcopy(row['X2'])
            row['X2'] = row['X1']
            row['X1'] = temp
        else:
            print('Aisa nahi ho sakta ')
        # print('After line\n', row)
        try:
            while abs(y1 - df.loc[index_v][1]) < 10:
                if index_v in df.index:
                    if abs(x1 - df.loc[index_v][0]) < diff and abs(y1 - df.loc[index_v][1]) < diff and abs(
                                    x2 - df.loc[index_v][2]) < diff and abs(y2 - df.loc[index_v][3]) < diff:
                        df = df.drop(index_v)
                    elif abs(y1 - df.loc[index_v][1]) < diff and (
                            x1 <= df.loc[index_v][0] and x2 >= df.loc[index_v][2]):
                        df = df.drop(index_v)           #TODO same line occurs twice
                    elif abs(y1 - df.loc[index_v][1]) < diff and (
                            x1 <= df.loc[index_v][0] and x2 >= df.loc[index_v][0]):
                        field_box.pop()
                        if x2 < df.iloc[index_v][2]:
                            field_box.append([x1, y1 - height, df.loc[index_v][2], y1, "Field", np.nan, 0])
                            x2 = df.loc[index_v][2]
                            df.loc[index_curr] = [x1, y1, df.loc[index_v][2], y1]
                            df = df.drop(index_v)
                        else:
                            df = df.drop(index_v)
                    elif abs(y1 - df.loc[index_v][1]) < diff and (
                            x1 >= df.loc[index_v][0] and x1 <= df.loc[index_v][2]):
                        field_box.pop()
                        if x2 > df.loc[index_v][2]:
                            field_box.append([df.loc[index_v][0], y1 - height, x2, y1, "Field", np.nan, 0])
                            x1 = df.loc[index_v][0]
                            df.loc[index_curr] = [df.loc[index_v][0], y1, x2, y1]
                            df = df.drop(index_v)
                        else:
                            field_box.append(
                                [df.loc[index_v][0], y1 - height, df.loc[index_v][2], y1, "Field", np.nan, 0])
                            df = df.drop(index_curr)
                    elif abs(y1 - df.loc[index_v][1]) < diff and abs(x2 - df.loc[index_v][0]) < diff:
                        field_box.pop()
                        field_box.append([x1, y1 - height, df.loc[index_v][2], y1, "Field", np.nan, 0])
                        x2 = df.loc[index_v][2]
                        df.loc[index_curr] = [x1, y1, df.loc[index_v][2], y1]
                        df = df.drop(index_v)
                    elif abs(y1 - df.loc[index_v][1]) < diff and abs(x1 - df.loc[index_v][2]) < diff:
                        field_box.pop()
                        field_box.append([df.loc[index_v][0], y1 - height, x2, y1, "Field", np.nan, 0])
                        df.loc[index_curr] = [df.loc[index_v][0], y1, x2, y1]
                        x1 = df.loc[index_v][0]
                        df = df.drop(index_v)
                    else:
                        pass
                index_v += 1
        except Exception as e:
            print(e)
        try:
            index_curr = index_r
            x1 = df.loc[index_r][0]
            y1 = df.loc[index_r][1]
            x2 = df.loc[index_r][2]
            y2 = df.loc[index_r][3]
            field_box.append([x1, y1 - height, x2, y2, "Field", np.nan, 0])
        except:
            pass

        #    """REMOVAL OF REMAINING DUPLICATE LINES"""
    cv2.imwrite("lines.jpg", img)
    img = img1.copy()
    df = df.sort_values(by=['Y1', 'X1']).reset_index(drop=True)
    for index_r,row in df.iterrows():
        index_v = index_r
        cv2.line(img, (row['X1'], row['Y1']), (row['X2'], row['Y2']), (0, 255, 0), 3)

        try:
            while abs(y1 - df.loc[index_v][1]) < 10:
                if index_v in df.index:
                    if abs(x1 - df.loc[index_v][0]) < diff and abs(y1 - df.loc[index_v][1]) < diff and abs(
                                    x2 - df.loc[index_v][2]) < diff and abs(y2 - df.loc[index_v][3]) < diff:
                        df = df.drop(index_v)
                    elif abs(y1 - df.loc[index_v][1]) < diff and (
                            x1 <= df.loc[index_v][0] and x2 >= df.loc[index_v][2]):
                        df = df.drop(index_v)
                    elif abs(y1 - df.loc[index_v][1]) < diff and (
                            x1 <= df.loc[index_v][0] and x2 >= df.loc[index_v][0]):
                        field_box.pop()
                        if x2 < df.iloc[index_v][2]:
                            field_box.append([x1, y1 - height, df.loc[index_v][2], y1, "Field", np.nan, 0])
                            df.loc[index_curr] = [x1, y1, df.loc[index_v][2], y1]
                            x2 = df.loc[index_v][2]
                            df = df.drop(index_v)
                        else:
                            df = df.drop(index_v)
                    elif abs(y1 - df.loc[index_v][1]) < diff and (
                            x1 >= df.loc[index_v][0] and x1 <= df.loc[index_v][2]):
                        field_box.pop()
                        if x2 > df.loc[index_v][2]:
                            field_box.append([df.loc[index_v][0], y1 - height, x2, y1, "Field", np.nan, 0])
                            df.loc[index_curr] = [df.loc[index_v][0], y1, x2, y1]
                            x1 = df.loc[index_v][0]
                            df = df.drop(index_v)
                        else:
                            field_box.append(
                                [df.loc[index_v][0], y1 - height, df.loc[index_v][2], y1, "Field", np.nan, 0])
                            df = df.drop(index_curr)
                    elif abs(y1 - df.loc[index_v][1]) < diff and abs(x2 - df.loc[index_v][0]) < diff:
                        field_box.pop()
                        field_box.append([x1, y1 - height, df.loc[index_v][2], y1, "Field", np.nan, 0])
                        df.loc[index_curr] = [x1, y1, df.loc[index_v][2], y1]
                        x2 = df.loc[index_v][2]
                        df = df.drop(index_v)
                    elif abs(y1 - df.loc[index_v][1]) < diff and abs(x1 - df.loc[index_v][2]) < diff:
                        field_box.pop()
                        field_box.append([df.loc[index_v][0], y1 - height, x2, y1, "Field", np.nan, 0])
                        df.loc[index_curr] = [df.loc[index_v][0], y1, x2, y1]
                        x1 = df.loc[index_v][0]
                        df = df.drop(index_v)
                    else:
                        pass
                index_v += 1
        except Exception as e:
            print(e)
        try:
            index_curr = index_r
            x1 = df.loc[index_r][0]
            y1 = df.loc[index_r][1]
            x2 = df.loc[index_r][2]
            y2 = df.loc[index_r][3]
            field_box.append([x1, y1 - height, x2, y2, "Field", np.nan, 0])
        except:
            pass
    cv2.imwrite("linesafter.jpg", img)
    img = img1.copy()
    for i, row in df.iterrows():
        cv2.line(img, (row['X1'], row['Y1']), (row['X2'], row['Y2']), (0, 255, 0), 3)
    cv2.imwrite("linesafter2.jpg", img)
    return df


def eliminate_duplicate_entry(df, df_box):
    diff = 30
    for ii,row in df_box.iterrows():
        for index_r,rows in df.iterrows():
            if ii in df_box.index and index_r in df.index:
                if abs(rows['X1'] - row['X1']) < diff and abs(rows['Y1'] - row['Y1']) < diff:
                    df = df.drop(index_r)
                elif abs(rows['X2'] - row['X2']) < diff and abs(rows['Y2'] - row['Y2']) < diff:
                    df = df.drop(index_r)
                elif abs(rows['X2'] - row['X2']) < diff and abs(rows['Y1'] - row['Y1']) < diff:
                    df = df.drop(index_r)
                elif abs(rows['X1'] - row['X1']) < diff and abs(rows['Y2'] - row['Y2']) < diff:
                    df = df.drop(index_r)
                elif rows['X1'] - row['X1'] < 10 and abs(rows['Y1'] - row['Y1'])<20 and  rows['X2']-row['X2']>-10:
                    print('line found on top border of box')
                    df = df.drop(index_r)
                elif rows['X1'] - row['X1'] < 10 and abs(rows['Y2'] - row['Y2'])<20 and  rows['X2']-row['X2']>-10:
                    print('line found on bottom border of box')
                    df = df.drop(index_r)
                else:
                    pass
    return df, df_box


def get_checkbox(df_box):
    checkbox = []
    for index_r,row in df_box.iterrows():
        w = abs(row['X2'] - row['X1'])
        if w < 80:
            checkbox.append(
                [row['X1'] - 5, row['Y1'] - 5, abs(row['Y2'] - row['Y1']) + 10, abs(row['X2'] - row['X1'] + 10), "checkbox", np.nan, 0])
            df_box = df_box.drop(index_r)
    return df_box, checkbox


def generate_label_box(data, height, img):
    text = []
    label_box = []
    value = ""
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h,_ = img.shape[:2]
    # print(img.shape)
    for j in range(len(data)):
        text.append(data[j].split(" "))
    data = ""
    X1, Y1 = int(text[0][1]), int(text[0][2])
    X2, Y2 = int(text[0][3]), int(text[0][4])

    for i in range(len(text)):
        data += text[i][0]
        w = int(text[i][3]) - int(text[i][1])
        y_dist = abs(Y2 - int(text[i][4]))
        x_dist = abs(X2 - int(text[i][1]))
        # print(text[i][0],text[i][1],text[i][2],text[i][3],text[i][4])
        if w < 40:
            if not (text[i][0].isalpha() or text[i][0].isdigit()):
                continue
            if y_dist < height and x_dist < height:
                X2, Y2 = int(text[i][3]), int(text[i][4])
                value += text[i][0]
            else:
                if len(value) > 1:
                    # print(value)
                    crop = img[h - Y2 - 5:h - Y1 + 5, X1 - 5:X2 + 5]
                    text1 = pytesseract.image_to_string(crop, lang='eng', config='--psm 7')
                    if not text1.isdigit():
                        # value = ""
                        # value +=text[i][0]
                        # continue
                        label_box.append([X1, Y2, abs(Y2 - Y1), abs(X2 - X1), "label", text1, "0"])
                        # cv2.imshow("crop", crop)
                        # cv2.waitKey(0)
                        # print(text1)

                X1, Y1 = int(text[i][1]), int(text[i][2])
                X2, Y2 = int(text[i][3]), int(text[i][4])
                value = ""
                value += text[i][0]
    crop = img[h - Y2 - 5:h - Y1 + 5, X1 - 5:X2 + 5]
    # cv2.imshow("crop", crop)
    # cv2.waitKey(0)
    text1 = pytesseract.image_to_string(crop, lang='eng', config='--psm 7')
    label_box.append([X1, Y2, abs(Y2 - Y1), abs(X2 - X1), "label", text1, "0"])
    # print(data)
    df = pd.DataFrame(label_box, columns=['X1', 'Y1', 'X2', 'Y2', 'Type', 'value', 'group'])
    return df
def reformation(img, rec_coordinate):
    coordinates = []
    for p,q,r,s in rec_coordinate:
        coordinates.append([p[0], p[1], q[0], q[1], r[0], r[1], s[0], s[1]])
    df = pd.DataFrame(coordinates,columns=['X1','Y1','X2','Y2','X3','Y3','X4','Y4'])

    df = eliminate_duplicate_box_eight(coordinates,30,img)
    df = df.sort_values(by=['Y1','X1']).reset_index(drop=True)
    start = df.iloc[0]
    img1 = img.copy()

    # cv2.circle(img1, (start['X1'], start['Y1']), 2, (0, 0, 255), 2)
    # cv2.circle(img1, (start['X2'], start['Y2']), 2, (0, 0, 255), 2)
    # cv2.circle(img1, (start['X3'], start['Y3']), 2, (0, 0, 255), 2)
    # cv2.circle(img1, (start['X4'], start['Y4']), 2, (0, 0, 255), 2)
    cv2.imwrite("error.jpg",img1)
    vertical = [[start['X1'],start['Y1']],[start['X2'],start['Y2']],[start['X3'],start['Y3']],[start['X4'],start['Y4']]]
    df_vertical = pd.DataFrame(vertical, columns = ['X','Y'])
    df_vertical = df_vertical.sort_values(by= ['X','Y']).reset_index(drop=True)
    vertical = df_vertical.values.tolist()
    # if abs(vertical[0][1]-vertical[2][1])<15
    if vertical[0][1] > vertical[1][1]:
        y = abs(vertical[1][1] - vertical[0][1])
        x = abs(vertical[0][0] - vertical[2][0])
        aspect = y/x
        src_img = [[vertical[1][0], vertical[1][1]], [vertical[3][0], vertical[3][1]], [vertical[0][0], vertical[0][1]]]
        dst_img = [[vertical[0][0], vertical[1][1]], [vertical[2][0], vertical[1][1]], [vertical[0][0], vertical[0][1]]]
    else:
        src_img = [[vertical[0][0], vertical[0][1]], [vertical[2][0], vertical[2][1]], [vertical[1][0], vertical[1][1]]]
        dst_img = [[vertical[0][0], vertical[2][1]], [vertical[2][0], vertical[2][1]], [vertical[0][0], vertical[1][1]]]
    img = extraction.transformation(img, src_img,dst_img)
    #cv2.imshow('transformed',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return start, img, aspect
    # box = [start['X1'],start['Y1'],start['X2'],start['Y2'],start['X3'],start['Y3'],start['X4'],start['Y4']]
    # h,w = img.shape[:2]
    # if abs(start['Y3']-start['Y1'])>h/2:
    #     if abs(start['X1']-start['X2'])>0:
    #         box[]


def detect_rectangles_eight(image):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # _, threshold = cv2.threshold(img, 240, 255, cv2.THRESH_BINARY)
    threshold = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2)
    #cv2.imshow("thr",threshold)
    cv2.waitKey(0)
    contours,_ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    coordinate = []
    # cnt = imutils.grab_contours(contours)
    # cnt = sorted(cnt, key=cv2.contourArea)
    # cnt.pop(0)
    # c = max(cnt, key=cv2.contourArea)
    # image1 = image.copy()
    # cv2.drawContours(image1, c,-1,(0,0,255),2)
    # cv2.imshow("contours", image1)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    #
    # for cnt in contours:
    #     c_area = cv2.contourArea(cnt)
    #     x,y,w,h = cv2.boundingRect(cnt)
    #     area = w*h
    #     if c_area/area>0.9:
    #         coordinate.append(([x,y+h],[x,y], [x+w,y+h],[x+w,y]))


    for cnt in contours:
        approx = cv2.approxPolyDP(cnt, 0.01 * cv2.arcLength(cnt, True), True)
        if len(approx) == 4 and cv2.contourArea(cnt) > 50:
            # if abs(approx[0][0][0] - approx[1][0][0]) < 10 or abs(approx[0][0][0] - approx[3][0][0]) < 10:
            coordinate.append((approx[0][0], approx[1][0], approx[2][0], approx[3][0]))
        #			'''if 2000+co > cv2.contourArea(cnt) > co:
        #				continue
        #				co=cv2.contourArea(cnt)'''# try changing the value in place of 2000 to get outer rectangles
        #
        # for i in range(len(coordinate)):
        #     cv2.rectangle(img, tuple(coordinate[i][0]), tuple(coordinate[i][1]), (0, 0, 255), 1)
        # cv2.circle(img, tuple(coordinate[i][0]), 5, (0, 255, 0), 5)
        # cv2.circle(img, tuple(coordinate[i][1]), 5, (0, 255, 0), 5)
    return img, coordinate

def eliminate_duplicate_box_eight(rec_coordinate, diff,img):
    # h,w = img.shape[:2]
    # w = 0.9*w
    # h = 0.9*h
    coordinates = []
    for x in rec_coordinate:
        coordinates.append([x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7]])
    df_box = pd.DataFrame(rec_coordinate, columns=['X1', 'Y1', 'X2', 'Y2', 'X3', 'Y3', 'X4', 'Y4'])
    df_box = df_box.sort_values(by=['Y1', 'X1']).reset_index(drop=True)
    index_curr, x1, y1, x2, y2 = 0, 0, 0, 0, 0
    for index_r,row in df_box.iterrows():
        index_v = index_r
        if row['X1']<row['X3']:
            if row['Y1']<row['Y3']:
                pass
            elif row['Y1']>row['Y3']:
                temp = copy.deepcopy(row['Y1'])
                # row['Y1'] = row['Y2']
                df_box.at[index_r,'Y1'] = copy.deepcopy(row['Y3'])
                # row['Y2']=temp
                df_box.at[index_r, 'Y3'] = temp
            else:
                print("unforeseen Condition")
        elif row['X1']>row['X3']:
            if row['Y1']<row['Y3']:
                temp = copy.deepcopy(row['X3'])
                row['X3'] = copy.deepcopy(row['X1'])
                df_box.at[index_r,'X3']= copy.deepcopy(row['X1'])
                df_box.at[index_r, 'X1'] = temp
            elif row['Y1']>row['Y3']:
                temp = copy.deepcopy(row['X3'])
                df_box.at[index_r, 'X3'] = copy.deepcopy(row['X1'])
                df_box.at[index_r, 'X3'] = temp
                temp = copy.deepcopy(row['Y3'])
                # row['Y2'] = row['Y1']
                df_box.at[index_r, 'Y3'] = copy.deepcopy(row['Y1'])
                df_box.at[index_r, 'X3'] = temp
            else:
                print("unforeseen Condition")
        if (row['X1'] < 10 and row['Y1'] < 10):  #(row['X1']>w and row['Y1']<10
            df_box = df_box.drop(index_r)
        try:
            while (index_v in df_box.index and abs(y1 - df_box.loc[index_v][1]) < 10):
                if index_v in df_box.index:
                    if abs(x1 - df_box.loc[index_v][0]) < diff and abs(y1 - df_box.loc[index_v][1]) < diff and abs(
                                    x2 - df_box.loc[index_v][2]) < diff and abs(y2 - df_box.loc[index_v][3]) < diff:
                        if x1 - df_box.loc[index_v][0] < 0:
                            df_box = df_box.drop(index_v)
                        else:
                            df_box = df_box.drop(index_curr)
                    else:
                        pass
                    index_v += 1
        except Exception as e:
            print(e)
        try:
            index_curr = index_r
            x1 = df_box.loc[index_r][0]
            y1 = df_box.loc[index_r][1]
            x2 = df_box.loc[index_r][2]
            y2 = df_box.loc[index_r][3]
        except:
            pass
    df_box = df_box.sort_values(by=['Y1', 'X1']).reset_index(drop=True)
    return df_box

# for number in range(20,23):
def reformation_filled_form(img, rec_coordinate, aspect):
    coordinates = []
    for p,q,r,s in rec_coordinate:
        coordinates.append([p[0], p[1], q[0], q[1], r[0], r[1], s[0], s[1]])
    df = pd.DataFrame(coordinates,columns=['X1','Y1','X2','Y2','X3','Y3','X4','Y4'])

    df = eliminate_duplicate_box_eight(coordinates,30,img)
    df = df.sort_values(by=['Y1','X1']).reset_index(drop=True)
    start = df.iloc[0]
    img1 = img.copy()

    # cv2.circle(img1, (start['X1'], start['Y1']), 2, (0, 0, 255), 2)
    # cv2.circle(img1, (start['X2'], start['Y2']), 2, (0, 0, 255), 2)
    # cv2.circle(img1, (start['X3'], start['Y3']), 2, (0, 0, 255), 2)
    # cv2.circle(img1, (start['X4'], start['Y4']), 2, (0, 0, 255), 2)
    cv2.imwrite("error.jpg",img1)
    vertical = [[start['X1'],start['Y1']],[start['X2'],start['Y2']],[start['X3'],start['Y3']],[start['X4'],start['Y4']]]
    df_vertical = pd.DataFrame(vertical, columns = ['X','Y'])
    df_vertical = df_vertical.sort_values(by= ['X','Y']).reset_index(drop=True)
    vertical = df_vertical.values.tolist()
    # if abs(vertical[0][1]-vertical[2][1])<15
    if vertical[0][1] > vertical[1][1]:
        y = abs(vertical[1][1]-vertical[0][1])
        x = int(y/aspect)
        src_img = [[vertical[1][0], vertical[1][1]], [vertical[3][0], vertical[3][1]], [vertical[0][0], vertical[0][1]]]
        dst_img = [[vertical[0][0], vertical[1][1]], [vertical[0][0]+x, vertical[1][1]], [vertical[0][0], vertical[0][1]]]
    else:
        src_img = [[vertical[0][0], vertical[0][1]], [vertical[2][0], vertical[2][1]], [vertical[1][0], vertical[1][1]]]
        dst_img = [[vertical[0][0], vertical[2][1]], [vertical[2][0], vertical[2][1]], [vertical[0][0], vertical[1][1]]]
    img = extraction.transformation(img, src_img,dst_img)
    #cv2.imshow('transformed',img)
    # cv2.waitKey(0)
    cv2.destroyAllWindows()
    return start, img
