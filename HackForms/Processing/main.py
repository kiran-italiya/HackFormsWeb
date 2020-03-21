# from . import *


import HackForms.Processing.extraction as extraction
import HackForms.Processing.detection as detection
import HackForms.Processing.hackForm as hackForm
# import extraction
# import detection

import cv2, os
import pytesseract
import imutils, copy, csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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
        cv2.imshow("crop", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # """Label Detection and Processing"""

        rectangle_image, rec_coordinate = detection.detect_rectangles_eight(img)
        start, img, aspect = detection.reformation(img, rec_coordinate)
        img = imutils.resize(img, width = 800)
        cv2.imwrite("cropped.jpg", img)


        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray, 3)
        threshold = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2)


        #    """RECTANGLE DETECTION"""


        cnt = detection.contour(img)

        rectangle_image, rec_coordinate = detection.detect_rectangles(img)

        df_box = detection.eliminate_duplicate_box(rec_coordinate, self.diff, img)
        cv2.imwrite("boxes.jpg", rectangle_image)

        # """Line detection and processing"""

        line = detection.linesp(img)

        df = detection.line_processing(line, self.diff, self.height, img)
        img2 = img.copy()
        for row in df.itertuples():
            cv2.rectangle(img2, (row[1], row[2] - 40), (row[3], row[4]), (0, 0, 255), 1)
        cv2.imwrite("lines.jpg", img2)
        # """Duplicate BOX/LINE ELIMINATION"""

        df, df_box = detection.eliminate_duplicate_entry(df, df_box)
        img1 = img.copy()
        print(df_box)

        for i, row in df.iterrows():
            cv2.rectangle(img1, (row['X1'], row['Y1']), (row['X2'], row['Y2']), (0, 255, 0), 2)
        cv2.imwrite("boxes.jpg", img1)
        #    """CHECKBOX DETECTION"""
        df_box, checkbox = detection.get_checkbox(df_box)
        # print('get checkbox  ==\n', df_box)
        # print("XXx", df_box)
        #    """FORM BOX """
        # start = df_box.iloc[0]
        # print(start)
        #    """CONVERSION TO H,W FROM X2,Y2"""
        df['Y1'] = df['Y1'] - self.height
        df['Y2'] = df['Y2'] + 5

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
        # print(df)
        # if (start[3]-start[1])>h/2:
        # 	threshold = threshold[start[1]:start[3],start[0]:start[2]]
        x, y = threshold[0:df.iloc[0]['top'] - 10, :].shape[:2]
        # print("=-=-===--==-=-=-=-=-=-=-=-=-=++",df.iloc[0]['top'])
        threshold[0:df.iloc[0]['top'] - 10, :] = np.ones((x, y)) * 255
        for row in df.itertuples():
            x, y = threshold[row[2] - pad:row[2] + row[3] + pad, row[1] - pad:row[1] + pad + row[4]].shape
            threshold[row[2] - pad:row[2] + row[3] + pad, row[1] - pad:row[1] + pad + row[4]] = np.ones((x, y)) * 255

        # cv2.imshow("img", threshold)
        # cv2.waitKey(0)
        cv2.destroyAllWindows()
        text = pytesseract.image_to_boxes(threshold, lang='eng', config='--psm 4')
        data = text.split('\n')
        label_height = 30
        df = detection.generate_label_box(data, label_height, img)

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
        img_new = img.copy()
        for _,row in df.iterrows():
            cv2.rectangle(img_new, (row['left'], row['top']), (row['left'] + row['width'], row['top'] + row['height']),(0, 0, 255), 2)
        cv2.imwrite('image_lines.jpg',img_new)
        return df, aspect

    def process_filled_form(self, df, img_name, df_final, aspect,overall_semantics,ix):

        # """INSERT LOOP FOR PROCESSING IMAGES IN BULK"""
        # img = cv2.imread("kiran2.png")
        img = cv2.imread(img_name)
        img = imutils.resize(img, width=1000)
        cv2.imshow("Test image", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        rectangle_image, rec_coordinate = detection.detect_rectangles_eight(img)
        dst_img, img = detection.reformation_filled_form(img, rec_coordinate, aspect)
        img = imutils.resize(img, width=800 )
        cv2.imwrite("croppedfilled.jpg",img)
        print(img.shape)
        img1 = img.copy()
        # for i, row in df.iterrows():
        #     cv2.rectangle(img1, (row['```left'],row['top']),(row['left']+row['width'], row['top']+row['height']), (0,0,255),2)
        # cv2.imwrite("boxeonimage.jpg",img1)
        dummy = [0] * len(df_final.columns)

        df_final.loc[len(df_final)] = dummy
        # print('final   ', df_final)
        length = len(df_final)
        df_final = extraction.radio_identification(img, df, df_final, length-1)
        df_final = extraction.checkbox_identification(img, df, df_final, length-1)
        # print('final   ',df_final)

        df_final,semantic = extraction.perform_OCR(img, df, df_final, length-1)
        overall_semantics+=semantic
        df_final.to_csv('final'+str(i)+'.csv')

        # dict = hackForm.data_dict(df, df_final)
        return df_final,overall_semantics  # , dict

    def processForm(self, img, path,i):
        df, aspect = self.process_empty_form(img_name=img)
        label = []
        label.clear()
        for _, row in df.iterrows():
            if row['type'] == 'label' and row['group'] == 'NaN':
                # print("This is a label ",row['value'])
                label.append(row['value'])

        df_final = pd.DataFrame(columns=label)
        # print(df_final)
        overall_form_semantics = 0
        for file in os.listdir(path):
            if file.endswith(".jpg"):
                tmp_df = df
                df_final,overall_form_semantics = self.process_filled_form(df=tmp_df, img_name=path + "/" + file, df_final=df_final, aspect=aspect,overall_semantics=overall_form_semantics,ix=i)
                print('~|||||||||||The overall semantics of this type of form is ||||||||||||||',overall_form_semantics)
            # self.database[file] = dict
        return df_final

    def generate_analytics(self):
        database = {
            '1.jpg': {
                "labels": ["Date", "Name", "Enrollmentno", "Whatdidyoulikeverymuch", "Attitudeofemployees",
                           "Environment", "Administration", "TalentandInnovation", "Infrastructure",
                           "Ratesupportivenessofemployees", "Ratetheimpactofprojectscompanytsdoing",
                           "Ratetheoverallexperienceofthevisit", "Ratequalityofproviddresources",
                           "Howmuchlikelydoyourecommendthecompanyfortheirservices", "Anucanetructiveenasactinne"],
                "fields": [{'Name': 'kireaii'}, {'Date': "19l01/2020"}, {'Enrollmentno', '1#/il13i4o'},
                           {'Anucanetructiveenasactinne': 'foof sdu not greal faciliifes were lakling entuciaim'}],
                "checkboxes": [
                    {
                        "Attitudeofemployees": {"Attitudeofemployees": 't', "Environment": 'f', "Administration": 't',
                                                "TalentandInnovation": 't', "Infrastructure": 'f'}
                    }
                ],
                "radios": {
                    "Ratesupportivenessofemployees": 4, "Ratetheimpactofprojectscompanytsdoing": 3,
                    "Ratetheoverallexperienceofthevisit": 5, "Ratequalityofproviddresources": 4,
                    "Howmuchlikelydoyourecommendthecompanyfortheirservices": 5
                },
            },
            '2.jpg': {
                "labels": ["Date", "Name", "Enrollmentno", "Whatdidyoulikeverymuch", "Attitudeofemployees",
                           "Environment", "Administration", "TalentandInnovation", "Infrastructure",
                           "Ratesupportivenessofemployees", "Ratetheimpactofprojectscompanytsdoing",
                           "Ratetheoverallexperienceofthevisit", "Ratequalityofproviddresources",
                           "Howmuchlikelydoyourecommendthecompanyfortheirservices", "Anucanetructiveenasactinne"],
                "fields": [{'Name': 'kireaii'}, {'Date': "19l01/2020"}, {'Enrollmentno', '1#/il13i4o'},
                           {'Anucanetructiveenasactinne': 'foof sdu not greal faciliifes were lakling entuciaim'}],
                "checkboxes": [
                    {
                        "Attitudeofemployees": {"Attitudeofemployees": 't', "Environment": 't', "Administration": 't',
                                                "TalentandInnovation": 't', "Infrastructure": 't'}
                    }
                ],
                "radios": {
                    "Ratesupportivenessofemployees": 2, "Ratetheimpactofprojectscompanytsdoing": 3,
                    "Ratetheoverallexperienceofthevisit": 2, "Ratequalityofproviddresources": 2,
                    "Howmuchlikelydoyourecommendthecompanyfortheirservices": 4
                },
            }
        }

        checkbox_df = None
        radio_df = None
        main_parents_list = []

        for file_name, file_dict in database.items():
            for main_parent_dict in file_dict['checkboxes']:
                for main_parent, dictionary in main_parent_dict.items():
                    main_parents_list.append(main_parent)
                    if checkbox_df is None:
                        dummy = [0] * (len(dictionary))
                        checkbox_df = pd.DataFrame([dummy], columns=dictionary.keys(), dtype=int)
                    print("cb_df::", checkbox_df)
                    print("dictionary:::",dictionary)
                    for parent, val in dictionary.items():
                        if val == 't':
                            checkbox_df.at[0, parent] += 1

            # if radio_df is None:
            #     radio_df = pd.DataFrame([[0] * (len(file_dict['radios']))], columns=file_dict['radios'].keys())
            # for parent, val in file_dict['radios'].items():
            #     radio_df.at[0, parent] += val



        # checkbox_plt = plt.figure()
        # labels = checkbox_df.columns
        # vals = checkbox_df.iloc[0]
        # plt.bar(labels, vals)
        # plt.xlabel("Categories")
        # plt.ylabel("values")
        # plt.show()
        # plot chart for checkboxes chart here label= parent   x-axis=parent.keys()  y-axis=parent.values()

        # radio_plt = plt.figure()
        # labels = radio_df.columns
        # vals = radio_df.iloc[0]
        # plt.bar(labels, vals)
        # plt.xlabel("Categories")
        # plt.ylabel("values")
        # plt.show()

    # plot chart for labels label=ratings x-axis=database['radios'].keys() y-axis=database['radios'].values()


pf = ProcessForm()
for i in range(4):
    if i!=2:
        pf.processForm('k'+str(i+1)+'.jpg' , os.path.join(os.getcwd(), 'data/k'+str(i+1)+'/'),i+1)

# pf.generate_analytics()


# ===================================================================
# using SendGrid's Python Library
# https://github.com/sendgrid/sendgrid-python
# import os
# from sendgrid import SendGridAPIClient
# from sendgrid.helpers.mail import Mail
#
# message = Mail(
#     from_email='from_email@example.com',
#     to_emails='to@example.com',
#     subject='Sending with Twilio SendGrid is Fun',
#     html_content='<strong>and easy to do anywhere, even with Python</strong>')
# try:
#     sg = SendGridAPIClient(os.environ.get('SENDGRID_API_KEY'))
#     response = sg.send(message)
#     print(response.status_code)
#     print(response.body)
#     print(response.headers)
# except Exception as e:
#     print(e.message)