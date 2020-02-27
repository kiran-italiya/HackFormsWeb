import cv2
import numpy as np
import pandas as pd
import extraction
import copy

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
            if radius > 8:
                if radius < 50:
                    if c_area > 0.8:
                        circles.append(
                            [int(x) - int(radius), int(y) - int(radius), int(2 * radius), int(2 * radius), "radio",
                             np.nan, 0])
                        cv2.circle(img, (int(x), int(y)), int(radius), (0, 0, 255), 1)
        cv2.imshow("contour", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        cv2.imwrite("circle.png", img)
        return circles
def eliminate_duplicate_circle(circles,diff,img):
    df_circle = pd.DataFrame(circles, columns=['X1', 'Y1', 'R1', 'R2',"type", "value", "group"])
    df_circle = df_circle.sort_values(by=['Y1', 'X1']).reset_index(drop=True)
    index_curr, x, y, r = 0, 0, 0, 0
    for row in df_circle.itertuples():
        index_v = row[0]
        print(row[1], row[0])
        if abs(x-row[1])<diff and abs(y- row[2])<diff:
            df_circle = df_circle.drop(row[0])
            while (index_v in df_circle.index and abs(y - df_circle.loc[index_v][1]) < 10):
                if index_v in df_circle.index:
                    if abs(x - df_circle.loc[index_v][0]) < diff and abs(y - df_circle.loc[index_v][1]) < diff:
                        if x - df_circle.loc[index_v][0] > 0:
                            df_circle = df_circle.drop(index_v)
                        else:
                            df_circle = df_circle.drop(index_curr)
                    else:
                        pass
                    index_v += 1
        try:
            index_curr = row[0]
            x = df_circle.loc[row[0]][0]
            y = df_circle.loc[row[0]][1]
            r = df_circle.loc[row[0]][2]
        except:
            pass
    df_circle = df_circle.sort_values(by=['Y1', 'X1']).reset_index(drop=True)
    return df_circle


def linesp(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    canny = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(canny, 1, np.pi / 180, 80, None, 50, 1)
    return lines


def detect_rectangles(image):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, threshold = cv2.threshold(img, 240, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    coordinate = []
    for cnt in contours:
        approx = cv2.approxPolyDP(cnt, 0.01 * cv2.arcLength(cnt, True), True)
        if len(approx) == 4 and cv2.contourArea(cnt) > 50:
            coordinate.append((approx[0][0], approx[2][0]))
        #			'''if 2000+co > cv2.contourArea(cnt) > co:
        #				continue
        #				co=cv2.contourArea(cnt)'''# try changing the value in place of 2000 to get outer rectangles

        for i in range(len(coordinate)):
            cv2.rectangle(img, tuple(coordinate[i][0]), tuple(coordinate[i][1]), (0, 0, 255), 1)
            cv2.circle(img, tuple(coordinate[i][0]), 5, (0, 255, 0), 5)
            cv2.circle(img, tuple(coordinate[i][1]), 5, (0, 255, 0), 5)
    return img, coordinate


def eliminate_duplicate_box(rec_coordinate, diff,img):
    h,w = img.shape[:2]
    w = 0.9*w
    h = 0.9*h
    coordinates = []
    for x, y in rec_coordinate:
        coordinates.append([x[0], x[1], y[0], y[1]])

    df_box = pd.DataFrame(coordinates, columns=['X1', 'Y1', 'X2', 'Y2'])
    df_box = df_box.sort_values(by=['Y1', 'X1']).reset_index(drop=True)
    index_curr, x1, y1, x2, y2 = 0, 0, 0, 0, 0
    for i,row in df_box.iterrows():
        index_v = i
        print('before  ',row
              )
        if row['X1']<row['X2']:
            if row['Y1']<row['Y2']:
                print('No changes  ')
                pass
            elif row['Y1']>row['Y2']:
                temp = copy.deepcopy(row['Y1'])
                # df_box.at(row[0row[2]=row[4]
                row['Y1'] = row['Y2']
                row['Y2']=temp
            else:
                print("unforeseen Condition")
        elif row['X1']>row['X2']:
            if row['Y1']<row['Y2']:
                temp = copy.deepcopy(row['X2'])
                row['X2'] = row['X1']
                row['X1'] = temp
            elif row['Y1']>row['Y2']:
                temp = copy.deepcopy(row['X2'])
                row['X2'] = row['X1']
                row['X1'] = temp
                temp = copy.deepcopy(row['Y2'])
                row['Y2'] = row['Y1']
                row['Y1'] = temp
                print('double trouble  ')
            else:
                print("unforeseen Condition")
        print('after   ', row)
        print(row['X1'], i)
        if (row['X1'] < 10 and row['Y1'] < 10) or (row['X1']>w and row['Y1']<10):
            df_box = df_box.drop(i)
        try:
            while (index_v in df_box.index and abs(y1 - df_box.loc[index_v][1]) < 10):
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
        try:
            index_curr = i
            x1 = df_box.loc[i][0]
            y1 = df_box.loc[i][1]
            x2 = df_box.loc[i][2]
            y2 = df_box.loc[i][3]
        except:
            pass
    df_box = df_box.sort_values(by=['Y1', 'X1']).reset_index(drop=True)
    return df_box


def line_processing(line, diff, height):
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

    #    """DUPLICATE LINES ELIMINATION"""
    for row in df.itertuples():
        index_v = row[0]
        try:
            while (index_v in df.index and abs(y1 - df.loc[index_v][1]) < 10):
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
            index_curr = row[0]
            x1 = df.loc[row[0]][0]
            y1 = df.loc[row[0]][1]
            x2 = df.loc[row[0]][2]
            y2 = df.loc[row[0]][3]
            field_box.append([x1, y1 - height, x2, y2, "Field", np.nan, 0])
        except:
            pass

        #    """REMOVAL OF REMAINING DUPLICATE LINES"""
    df = df.sort_values(by=['Y1', 'X1']).reset_index(drop=True)
    for row in df.itertuples():
        index_v = row[0]

        try:
            while (index_v in df.index and abs(y1 - df.loc[index_v][1]) < 10):
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
            index_curr = row[0]
            x1 = df.loc[row[0]][0]
            y1 = df.loc[row[0]][1]
            x2 = df.loc[row[0]][2]
            y2 = df.loc[row[0]][3]
            field_box.append([x1, y1 - height, x2, y2, "Field", np.nan, 0])
        except:
            pass
    return df


def eliminate_duplicate_entry(df, df_box):
    diff = 30
    for row in df_box.itertuples():
        for rows in df.itertuples():
            if row[0] in df_box.index and rows[0] in df.index:
                if abs(row[1] - rows[1]) < diff and abs(row[2] - rows[2]) < diff:
                    df = df.drop(rows[0])
                elif abs(row[3] - rows[3]) < diff and abs(row[4] - rows[4]) < diff:
                    df = df.drop(rows[0])
                elif abs(row[3] - rows[3]) < diff and abs(row[2] - rows[2]) < diff:
                    df = df.drop(rows[0])
                elif abs(row[1] - rows[1]) < diff and abs(row[4] - rows[4]) < diff:
                    df = df.drop(rows[0])
                else:
                    pass
    return df, df_box


def get_checkbox(df_box):
    checkbox = []
    for row in df_box.itertuples():
        w = abs(row[3] - row[1])
        if w < 80:
            checkbox.append(
                [row[1] - 5, row[2] - 5, abs(row[4] - row[2]) + 10, abs(row[3] - row[1] + 10), "checkbox", np.nan, 0])
            df_box = df_box.drop(row[0])
    return df_box, checkbox


def generate_label_box(data, height):
    text = []
    label_box = []
    value = ""

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
        if w < 40:
            if not text[i][0].isalpha():
                continue
            if y_dist < height and x_dist < height:
                X2, Y2 = int(text[i][3]), int(text[i][4])
                value += text[i][0]
            else:
                if len(value) > 1:
                    if (Y2 - Y1) < 25:
                        label_box.append([X1, Y1, 25, X2 - X1, "label", value, "0"])
                    else:
                        label_box.append([X1, Y1, Y2 - Y1, X2 - X1, "label", value, "0"])
                X1, Y1 = int(text[i][1]), int(text[i][2])
                X2, Y2 = int(text[i][3]), int(text[i][4])
                value = ""
                value += text[i][0]
    label_box.append([X1, Y1, Y2 - Y1, X2 - X1, "label", value, "0"])
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
    vertical = [[start['X1'],start['Y1']],[start['X2'],start['Y2']],[start['X3'],start['Y3']],[start['X4'],start['Y4']]]
    df_vertical = pd.DataFrame(vertical, columns = ['X','Y'])
    df_vertical = df_vertical.sort_values(by= ['X','Y']).reset_index(drop=True)
    vertical = df_vertical.values.tolist()
    src_img = [[vertical[0][0],vertical[0][1]],[vertical[2][0],vertical[2][1]],[vertical[1][0],vertical[1][1]]]
    dst_img = [[vertical[0][0],vertical[0][1]],[vertical[2][0],vertical[0][1]],[vertical[0][0],vertical[1][1]]]
    img = extraction.transformation(img, src_img,dst_img)
    cv2.imshow('transformed',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return start, img
    # box = [start['X1'],start['Y1'],start['X2'],start['Y2'],start['X3'],start['Y3'],start['X4'],start['Y4']]
    # h,w = img.shape[:2]
    # if abs(start['Y3']-start['Y1'])>h/2:
    #     if abs(start['X1']-start['X2'])>0:
    #         box[]


def detect_rectangles_eight(image):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, threshold = cv2.threshold(img, 240, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    coordinate = []
    for cnt in contours:
        approx = cv2.approxPolyDP(cnt, 0.01 * cv2.arcLength(cnt, True), True)
        if len(approx) == 4 and cv2.contourArea(cnt) > 50:
            coordinate.append((approx[0][0],approx[1][0],approx[2][0],approx[3][0]))
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
    h,w = img.shape[:2]
    w = 0.9*w
    h = 0.9*h
    coordinates = []
    for x in rec_coordinate:
        coordinates.append([x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7]])
    df_box = pd.DataFrame(rec_coordinate, columns=['X1', 'Y1', 'X2', 'Y2', 'X3', 'Y3', 'X4', 'Y4'])
    df_box = df_box.sort_values(by=['Y1', 'X1']).reset_index(drop=True)
    index_curr, x1, y1, x2, y2 = 0, 0, 0, 0, 0
    for row in df_box.itertuples():
        index_v = row[0]
        if (row[1] < 10 and row[2] < 10) or (row[1]>w and row[2]<10):
            df_box = df_box.drop(row[0])
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
            index_curr = row[0]
            x1 = df_box.loc[row[0]][0]
            y1 = df_box.loc[row[0]][1]
            x2 = df_box.loc[row[0]][2]
            y2 = df_box.loc[row[0]][3]
        except:
            pass
    df_box = df_box.sort_values(by=['Y1', 'X1']).reset_index(drop=True)
    return df_box

# for number in range(20,23):
