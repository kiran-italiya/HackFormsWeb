import detection
import hackForm
import extraction
import cv2, os
import pytesseract
import imutils, copy, csv
import pandas as pd
import numpy as np


class ProcessForm:
    def __init__(self):

        self.width = 1000
        self.height = 40
        self.diff = 30

        self.img = cv2.imread("kiran.jpg")
        self.datacsv = 'data.csv'

        self.start = 0
        self.database = {}

    def process_empty_form(self, img_name):
        img = cv2.imread(img_name)
        img = imutils.resize(img, width=self.width)
        h, w = img.shape[:2]
        img1 = img.copy()
        cv2.imshow("crop", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, threshold = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
        gray = cv2.medianBlur(gray, 3)
        threshold = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2)
        gray = cv2.erode(threshold, None)
        gray = cv2.dilate(gray, None)

        # """Label Detection and Processing"""

        rectangle_image, rec_coordinate = detection.detect_rectangles_eight(img)
        start, img = detection.reformation(img, rec_coordinate)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray, 3)
        threshold = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2)
        gray = cv2.erode(threshold, None)
        gray = cv2.dilate(gray, None)

        # """Label Detection and Processing"""


        #    """RECTANGLE DETECTION"""


        cnt = detection.contour(img)

        rectangle_image, rec_coordinate = detection.detect_rectangles(img)
        df_box = detection.eliminate_duplicate_box(rec_coordinate, self.diff, img)
        print(df_box)
        cv2.imwrite("boxes.jpg", rectangle_image)

        # """Line detection and processing"""

        line = detection.linesp(img)
        df = detection.line_processing(line, self.diff, self.height)
        img2 = img.copy()
        for row in df.itertuples():
            cv2.rectangle(img2, (row[1], row[2] - 40), (row[3], row[4]), (0, 0, 255), 1)
        cv2.imwrite("lines.jpg", img2)
        # """Duplicate BOX/LINE ELIMINATION"""
        df, df_box = detection.eliminate_duplicate_entry(df, df_box)
        #    """CHECKBOX DETECTION"""
        df_box, checkbox = detection.get_checkbox(df_box)
        print("XXx", df_box)
        #    """FORM BOX """
        # start = df_box.iloc[0]
        # print(start)
        #    """CONVERSION TO H,W FROM X2,Y2"""
        df['Y1'] = df['Y1'] - self.height
        df['Y2'] = df['Y2'] + 10

        df_box = df_box.append(df)
        df_box = df_box.sort_values(by=['Y1', 'X1']).reset_index(drop=True)
        df_box['Type'] = 'field'
        df_box['Value'] = np.nan
        df_box['Group'] = 0
        temp1 = copy.deepcopy(df_box['X1'])
        temp2 = copy.deepcopy(df_box['X2'])
        temp3 = copy.deepcopy(df_box['Y1'])
        temp4 = copy.deepcopy(df_box['Y2'])

        df_box['Y2'] = abs(temp2 - temp1)
        df_box['X2'] = abs(temp4 - temp3)
        df_box['Y1'] = df_box['Y1'] - 5
        df_box['X2'] = df_box['X2'] + 10

        # """ADJUSTING COORDINATES TO BOUNDING BOX"""

        # if (start[3]-start[1])>h/2:
        # 	df_box = df_box.drop(df_box.index[0])


        field_box = df_box.values.tolist()

        circles = detection.findCircle(cnt, img)
        circles = detection.eliminate_duplicate_circle(circles, self.diff, img)
        circles = circles.values.tolist()
        with open(self.datacsv, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["left", "top", "height", "width", "type", "value", "group"])
            writer.writerows(field_box)
            writer.writerows(circles)
            writer.writerows(checkbox)

        pad = 10
        df = pd.read_csv(self.datacsv)
        print(df)
        # if (start[3]-start[1])>h/2:
        # 	threshold = threshold[start[1]:start[3],start[0]:start[2]]
        x, y = threshold[0:df.iloc[0]['top'] - 30, :].shape[:2]
        threshold[0:df.iloc[0]['top'] - 30, :] = np.ones((x, y)) * 255
        for row in df.itertuples():
            x, y = threshold[row[2] - pad:row[2] + row[3] + pad, row[1] - pad:row[1] + pad + row[4]].shape
            threshold[row[2] - pad:row[2] + row[3] + pad, row[1] - pad:row[1] + pad + row[4]] = np.ones((x, y)) * 255

        cv2.imshow("img", threshold)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        text = pytesseract.image_to_boxes(threshold, lang='eng', config='--psm 4')
        data = text.split('\n')
        label_height = 25
        df = detection.generate_label_box(data, label_height)

        y, x = threshold.shape
        df['Y1'] = y - 25 - df['Y1']
        df['X1'] = df['X1'] - 5
        df['X2'] = df['X2'] + 10
        df['Y2'] = df['Y2'] + 10

        # if (start[3]-start[1])>h/2:
        # 	df['Y1'] = df['Y1'] - start[1]
        # 	df['X1'] = df['X1'] - start[0]



        label_box = df.values.tolist()

        with open(self.datacsv, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(label_box)
        df = hackForm.hackForm(self.datacsv)
        return df

    def process_filled_form(self, df, img_name, df_final):

        # """INSERT LOOP FOR PROCESSING IMAGES IN BULK"""
        # img = cv2.imread("kiran2.png")
        img = cv2.imread(img_name)
        img = imutils.resize(img, width=1000)
        cv2.imshow("Test image", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        rectangle_image, rec_coordinate = detection.detect_rectangles_eight(img)
        dst_img, img = detection.reformation(img, rec_coordinate)
        cv2.imshow("boxes", rectangle_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        dummy = [0] * len(df_final.columns)
        print('dummy=',dummy)
        # df_final = df_final.append()
        df_final.loc[len(df_final)] = dummy
        print('final   ', df_final)
        df_final = extraction.radio_identification(img, df, df_final)
        df_final = extraction.checkbox_identification(img, df, df_final)
        print('final   ',df_final)
        df_final.to_csv('final.csv')

        dict = hackForm.data_dict(df, df_final)
        return df_final, dict

    def processForm(self, img, path):

        df = self.process_empty_form(img_name=img)
        label = []
        label.clear()
        for _, row in df.iterrows():
            if row['type'] == 'label' and row['group'] == 'NaN':
                print("This is a label ",row['value'])
                label.append(row['value'])

        df_final = pd.DataFrame(columns=label)
        print(df_final)
        for file in os.listdir(path):
            if file.endswith(".jpg"):
                tmp_df = df
                df_final, dict = self.process_filled_form(df=tmp_df, img_name=path + "/" + file, df_final=df_final)
            self.database[file] = dict
        print(df_final)


pf = ProcessForm()
pf.processForm('img_real.jpg', os.path.join(os.getcwd(), "data"))


def data_analytics(database):
    x = ''
    for file, file_data in database.values():
        labels = file_data["labels"]
        checkboxes = file_data["checkboxes"]
        radio = file_data["radio"]
        fields = file_data["fields"]

    return x
