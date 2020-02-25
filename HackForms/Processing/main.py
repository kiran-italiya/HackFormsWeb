import detection 
#import hackForm
#import extraction
import cv2
import pytesseract
import imutils,copy,csv
import pandas as pd
import numpy as np

img = cv2.imread("Test1.jpg")
print("test.png")
width = 1000
height = 40
diff = 30
img = imutils.resize(img, width=width)
img3 = img.copy()
img1 = img.copy()
img2 = img.copy()
cv2.imshow("crop", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, threshold = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
gray = cv2.medianBlur(gray, 3)
threshold = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY, 31,2)
gray = cv2.erode(threshold, None)
gray = cv2.dilate(gray, None)

#"""Label Detection and Processing"""

cnt = detection.contour(img1)

#    """RECTANGLE DETECTION"""

rectangle_image,rec_coordinate=detection.detect_rectangles(img)
df_box = detection.eliminate_duplicate_box(rec_coordinate,diff)

#"""Line detection and processing"""    
line = detection.linesp(img)
df = detection.line_processing(line,diff)
img2 = img.copy()
for row in df.itertuples():
	cv2.rectangle(img2, (row[1],row[2]),(row[3],row[4]),(0,0,255),1)
cv2.imwrite("lines.jpg",img2)
#"""Duplicate BOX/LINE ELIMINATION"""   
df,df_box = detection.eliminate_duplicate_entry(df,df_box)
#    """CHECKBOX DETECTION"""
df_box, checkbox = detection.get_checkbox(df_box)

#    """FORM BOX """
start = df_box.iloc[0]
print(start)
#    """CONVERSION TO H,W FROM X2,Y2"""
df['Y1'] = df['Y1'] - height
df['Y2'] = df['Y2'] + 10
df_box = df_box.drop(df_box.index[0])
df_box = df_box.append(df)
df_box = df_box.sort_values(by= ['Y1','X1']).reset_index(drop=True)
df_box['Type'] = 'field'
df_box['Value'] = np.nan
df_box['Group'] = 0
temp1 = copy.deepcopy(df_box['X1'])
temp2 = copy.deepcopy(df_box['X2'])
temp3 = copy.deepcopy(df_box['Y1'])
temp4 = copy.deepcopy(df_box['Y2'])

df_box['Y2'] = abs(temp2-temp1)
df_box['X2'] = abs(temp4-temp3)
df_box['Y1'] = df_box['Y1'] - 5
df_box['X2'] = df_box['X2'] + 10
field_box = df_box.values.tolist()

circles = detection.findCircle(cnt, img1)

cv2.imwrite("img.png",img1)
with open('data.csv', 'w', newline='') as file:
	writer = csv.writer(file)
	writer.writerow(["left", "top", "height", "width", "type", "value", "group"])
	writer.writerows(field_box)
	writer.writerows(circles)
	writer.writerows(checkbox)

pad = 10
df = pd.read_csv('data.csv')

for row in df.itertuples():
    x,y = threshold[row[2]-pad:row[2]+row[3]+pad,row[1]-pad:row[1]+pad+row[4]].shape
    threshold[row[2]-pad:row[2]+row[3]+pad,row[1]-pad:row[1]+pad+row[4]] = np.ones((x,y))*255




cv2.imshow("img",threshold)
cv2.waitKey(0)
cv2.destroyAllWindows()
text = pytesseract.image_to_boxes(threshold, lang = 'eng',config= '--psm 4')
data = text.split('\n')
height= 25
df = detection.generate_label_box(data,height)

y,x = threshold.shape
df['Y1'] = y - 25 - df['Y1']
df['X1'] = df['X1'] - 5
df['X2'] = df['X2'] + 10
df['Y2'] = df['Y2'] + 10
label_box = df.values.tolist()

with open('data.csv', 'a', newline='') as file:
	writer = csv.writer(file)
	writer.writerows(label_box)

df_box= pd.read_csv('data.csv',     encoding = "ISO-8859-1")
#df_box['top'] = df_box['top']-start[1]
#df_box['left'] = df_box['left']-start[0]
#img1 = img1[start[1]:start[3],start[0]:start[2]]
for row in df_box.itertuples():
	cv2.rectangle(img1, (row[1],row[2]),(row[1]+row[4],row[2]+row[3]),(0,0,255),1)
cv2.imshow("Display", img1)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite("result.jpg", img1)
