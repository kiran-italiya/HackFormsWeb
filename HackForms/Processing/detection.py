import cv2
import numpy as np
import pytesseract
import imutils 
import csv
import pandas as pd
import copy


def contour(image):
	img=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
	_, threshold = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY)
	_,contours,_=cv2.findContours(threshold,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
	return contours

def findCircle(cnts,img1):
	if len(cnts) == 0:
		print("No contour found!!")
	else:
		img = img1.copy()
		circles = []
		# it to compute the minimum enclosing circle andSub1v4V
		for c in cnts:
			((x, y), radius) = cv2.minEnclosingCircle(c)
			c_area = cv2.contourArea(c)
			c_area = c_area/((3.14)*radius*radius)
			if radius>8:
				if radius<50:
					if c_area>0.8:
						circles.append([int(x)-int(radius), int(y)-int(radius), int(2*radius),int(2*radius), "radio", np.nan, 0])
						cv2.circle(img, (int(x), int(y)), int(radius),(0,0,255), 1)
		cv2.imshow("contour", img)
		cv2.waitKey(0)
		cv2.destroyAllWindows()
		cv2.imwrite("circle.png",img)
		return circles

def linesp(img):
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	canny = cv2.Canny(gray, 50,150, apertureSize = 3)
	lines = cv2.HoughLinesP(canny, 1, np.pi/180, 80,None, 50, 1)
	return lines
def detect_rectangles(image):

	img=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
	_, threshold = cv2.threshold(img, 240, 255, cv2.THRESH_BINARY)
	_,contours,_=cv2.findContours(threshold,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
	coordinate=[]
	for cnt in contours:

		approx = cv2.approxPolyDP(cnt, 0.01*cv2.arcLength(cnt, True), True)
		if len(approx) == 4 and cv2.contourArea(cnt)>100:
			'''if 2000+co > cv2.contourArea(cnt) > co:         
				continue
				co=cv2.contourArea(cnt)'''# try changing the value in place of 2000 to get outer rectangles
			coordinate.append((approx[0][0],approx[2][0]))
	for i in range(len(coordinate)):
		cv2.circle(img,tuple(coordinate[i][0]),5,(0,255,0),5)
		cv2.circle(img,tuple(coordinate[i][1]),5,(0,255,0),5)
	return img,coordinate
def eliminate_duplicate_box(rec_coordinate,diff):
	coordinates = []
	for x,y in rec_coordinate:
		coordinates.append([x[0],x[1],y[0],y[1]])
	df_box = pd.DataFrame(coordinates, columns = ['X1','Y1','X2','Y2'])

	df_box = df_box.sort_values(by= ['Y1','X1']).reset_index(drop=True)
	index_curr,x1,y1,x2,y2 = 0,0,0,0,0
	for row in df_box.itertuples():
		index_v = row[0]
		if row[1]<10 and row[2]<10:
			df_box = df_box.drop(row[0])
		try:
			while(index_v in df_box.index and abs(y1-df_box.loc[index_v][1])<10):
				if index_v in df_box.index:
					if abs(x1-df_box.loc[index_v][0])<diff and abs(y1-df_box.loc[index_v][1])<diff and abs(x2-df_box.loc[index_v][2])<diff and abs(y2-df_box.loc[index_v][3])<diff:
						if x1-df_box.loc[index_v][0]<0:
							df_box = df_box.drop(index_v)
						else:
							df_box = df_box.drop(index_curr)
					else:
						pass
					index_v+=1
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
	df_box = df_box.sort_values(by= ['Y1','X1']).reset_index(drop=True)
	return df_box
def line_processing(line,diff):
	horiz_lines = []
	if line is not None:
		for j in range(0, len(line)):
			l = line[j][0]
			if abs(l[1]-l[3])<diff:
				horiz_lines.append(line[j][0])

	df = pd.DataFrame(horiz_lines, columns=['X1','Y1','X2','Y2'])
	df = df.sort_values(by= ['Y1','X1']).reset_index(drop=True)
	x1,y1,x2,y2 = 0,0,0,0
	field_box = []
	index_curr = 0


	for row in horiz_lines:
		cv2.line(img2,(row[0],row[1]),(row[2],row[3]),(0,0,255),2)

#    """DUPLICATE LINES ELIMINATION"""
	for row in df.itertuples():
		index_v = row[0]
		try:
			while(index_v in df.index and abs(y1-df.loc[index_v][1])<10):
				if index_v in df.index:
					if abs(x1-df.loc[index_v][0])<diff and abs(y1-df.loc[index_v][1])<diff and abs(x2-df.loc[index_v][2])<diff and abs(y2-df.loc[index_v][3])<diff:
						df = df.drop(index_v)
					elif abs(y1-df.loc[index_v][1])<diff and (x1<=df.loc[index_v][0] and x2>=df.loc[index_v][2]):
						df = df.drop(index_v)
					elif abs(y1-df.loc[index_v][1])<diff and (x1<=df.loc[index_v][0] and x2>=df.loc[index_v][0]):
						field_box.pop()
						if x2<df.iloc[index_v][2]:
							field_box.append([x1, y1-height, df.loc[index_v][2], y1, "Field", np.nan, 0])
							x2 = df.loc[index_v][2]
							df.loc[index_curr] = [x1, y1, df.loc[index_v][2], y1]
							df = df.drop(index_v)
						else:
							df = df.drop(index_v)
					elif abs(y1-df.loc[index_v][1])<diff and (x1>=df.loc[index_v][0] and x1<=df.loc[index_v][2]):
						field_box.pop()
						if x2>df.loc[index_v][2]:
							field_box.append([df.loc[index_v][0], y1-height, x2, y1, "Field", np.nan, 0])
							x1 = df.loc[index_v][0]
							df.loc[index_curr] = [df.loc[index_v][0], y1, x2, y1]
							df = df.drop(index_v)
						else:
							field_box.append([df.loc[index_v][0], y1-height, df.loc[index_v][2], y1, "Field", np.nan, 0])
							df = df.drop(index_curr)
					elif abs(y1-df.loc[index_v][1])<diff and abs(x2-df.loc[index_v][0])<diff:
						field_box.pop()
						field_box.append([x1, y1-height, df.loc[index_v][2], y1, "Field", np.nan, 0])
						x2 = df.loc[index_v][2]
						df.loc[index_curr] = [x1, y1, df.loc[index_v][2], y1]
						df = df.drop(index_v)
					elif abs(y1-df.loc[index_v][1])<diff and abs(x1-df.loc[index_v][2])<diff:
						field_box.pop()
						field_box.append([df.loc[index_v][0], y1-height, x2, y1, "Field", np.nan, 0])
						df.loc[index_curr]= [df.loc[index_v][0], y1, x2, y1]
						x1 = df.loc[index_v][0]
						df = df.drop(index_v)
					else:
						pass
				index_v+=1
		except Exception as e:
			print(e)
		try:
			index_curr = row[0]
			x1 = df.loc[row[0]][0]
			y1 = df.loc[row[0]][1]
			x2 = df.loc[row[0]][2]
			y2 = df.loc[row[0]][3]
			field_box.append([x1,y1-height,x2,y2,"Field",np.nan,0])
		except:
			pass

#    """REMOVAL OF REMAINING DUPLICATE LINES"""
	df = df.sort_values(by= ['Y1','X1']).reset_index(drop=True)
	for row in df.itertuples():
		index_v = row[0]

		try:
			while(index_v in df.index and abs(y1-df.loc[index_v][1])<10):
				if index_v in df.index:
					if abs(x1-df.loc[index_v][0])<diff and abs(y1-df.loc[index_v][1])<diff and abs(x2-df.loc[index_v][2])<diff and abs(y2-df.loc[index_v][3])<diff:
						df = df.drop(index_v)
					elif abs(y1-df.loc[index_v][1])<diff and (x1<=df.loc[index_v][0] and x2>=df.loc[index_v][2]):
						df = df.drop(index_v)
					elif abs(y1-df.loc[index_v][1])<diff and (x1<=df.loc[index_v][0] and x2>=df.loc[index_v][0]):
						field_box.pop()
						if x2<df.iloc[index_v][2]:
							field_box.append([x1, y1-height, df.loc[index_v][2], y1, "Field", np.nan, 0])
							df.loc[index_curr] = [x1, y1, df.loc[index_v][2], y1]
							x2 = df.loc[index_v][2]
							df = df.drop(index_v)
						else:
							df = df.drop(index_v)
					elif abs(y1-df.loc[index_v][1])<diff and (x1>=df.loc[index_v][0] and x1<=df.loc[index_v][2]):
						field_box.pop()
						if x2>df.loc[index_v][2]:
							field_box.append([df.loc[index_v][0], y1-height, x2, y1, "Field", np.nan, 0])
							df.loc[index_curr] = [df.loc[index_v][0], y1, x2, y1]
							x1 = df.loc[index_v][0]
							df = df.drop(index_v)
						else:
							field_box.append([df.loc[index_v][0], y1-height, df.loc[index_v][2], y1, "Field", np.nan, 0])
							df = df.drop(index_curr)
					elif abs(y1-df.loc[index_v][1])<diff and abs(x2-df.loc[index_v][0])<diff:
						field_box.pop()
						field_box.append([x1, y1-height, df.loc[index_v][2], y1, "Field", np.nan, 0])
						df.loc[index_curr] = [x1, y1, df.loc[index_v][2], y1]
						x2 = df.loc[index_v][2]
						df = df.drop(index_v)
					elif abs(y1-df.loc[index_v][1])<diff and abs(x1-df.loc[index_v][2])<diff:
						field_box.pop()
						field_box.append([df.loc[index_v][0], y1-height, x2, y1, "Field", np.nan, 0])
						df.loc[index_curr]= [df.loc[index_v][0], y1, x2, y1]
						x1 = df.loc[index_v][0]
						df = df.drop(index_v)
					else:
						pass
				index_v+=1
		except Exception as e:
			print(e)
		try:
			index_curr = row[0]
			x1 = df.loc[row[0]][0]
			y1 = df.loc[row[0]][1]
			x2 = df.loc[row[0]][2]
			y2 = df.loc[row[0]][3]
			field_box.append([x1,y1-height,x2,y2,"Field",np.nan,0])
		except:
			pass
	return df
def eliminate_duplicate_entry(df,df_box):
	count = 0
	for row in df_box.itertuples():
		for rows in df.itertuples():
			if row[0] in df_box.index and rows[0] in df.index:
				if abs(row[1]-rows[1])<diff and abs(row[2]-rows[2])<diff:
					df = df.drop(rows[0])
					count+=1
				elif abs(row[3]-rows[3])<diff and abs(row[4]-rows[4])<diff:
					df = df.drop(rows[0])
					count+=1
				elif abs(row[3]-rows[3])<diff and abs(row[2]-rows[2])<diff:
					df = df.drop(rows[0])
					count+=1
				elif abs(row[1]-rows[1])<diff and abs(row[4]-rows[4])<diff:
					df = df.drop(rows[0])
					count+=1
				else:
					pass
	return df,df_box
def get_checkbox(df_box):
	checkbox = []
	for row in df_box.itertuples():
		w = row[3] - row[1]
		if w < 80:
			checkbox.append([row[1]-5,row[2]-5,abs(row[4]-row[2])+10,abs(row[3]-row[1]+10),"checkbox",np.nan,0])
			df_box = df_box.drop(row[0])
	return df_box,checkbox
def generate_label_box(data,height):
	text = []
	label_box = []
	value = ""



	for j in range(len(data)):
		text.append(data[j].split(" "))
	data = ""
	X1,Y1 = int(text[0][1]),int(text[0][2])
	X2,Y2 = int(text[0][3]),int(text[0][4])
	for i in range(len(text)):
		data+=text[i][0]
		w = int(text[i][3])-int(text[i][1])
		y_dist = abs(Y2-int(text[i][4]))
		x_dist = abs(X2-int(text[i][1]))
		if w<40:
			if text[i][0].isdigit():
				continue
			if y_dist<height and x_dist<height:
				X2,Y2 = int(text[i][3]),int(text[i][4])
				value+= text[i][0]
			else:
				if len(value)>1:
					if (Y2-Y1)<25:
						label_box.append([X1,Y1,25,X2-X1,"label",value,"0"])
					else:
						label_box.append([X1,Y1,Y2-Y1,X2-X1,"label",value,"0"])
				X1,Y1 = int(text[i][1]),int(text[i][2])
				X2,Y2 = int(text[i][3]),int(text[i][4])
				value = ""
				value+=text[i][0]
	label_box.append([X1,Y1,Y2-Y1,X2-X1,"label",value,"0"])
	df = pd.DataFrame(label_box, columns = ['X1','Y1','X2','Y2','Type','value','group'])
	return df

# for number in range(20,23):
img = cv2.imread("test21.jpg")
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

cnt = contour(img1)

#    """RECTANGLE DETECTION"""

rectangle_image,rec_coordinate=detect_rectangles(img)
df_box = eliminate_duplicate_box(rec_coordinate,diff)

#"""Line detection and processing"""    
line = linesp(img)
df = line_processing(line,diff)


#"""Duplicate BOX/LINE ELIMINATION"""   
df,df_box = eliminate_duplicate_entry(df,df_box)
#    """CHECKBOX DETECTION"""
df_box, checkbox = get_checkbox(df_box)

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

circles = findCircle(cnt, img1)

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
	threshold[row[2]-pad:row[2]+row[3]+pad,row[1]-pad:row[1]+pad+row[4]] = np.ones((row[3]+(2*pad),row[4]+(2*pad)))*255




cv2.imshow("img",threshold)
cv2.waitKey(0)
cv2.destroyAllWindows()
text = pytesseract.image_to_boxes(threshold, config= '--psm 1')
data = text.split('\n')
height= 25
df = generate_label_box(data,height)

y,x = threshold.shape
df['Y1'] = y - 25 - df['Y1']
df['X1'] = df['X1'] - 5
df['X2'] = df['X2'] + 10
df['Y2'] = df['Y2'] + 10
label_box = df.values.tolist()

with open('data.csv', 'a', newline='') as file:
	writer = csv.writer(file)
	writer.writerows(label_box)

df_box= pd.read_csv('data.csv')
df_box['top'] = df_box['top']-start[1]
df_box['left'] = df_box['left']-start[0]
img1 = img1[start[1]:start[3],start[0]:start[2]]
for row in df_box.itertuples():
	cv2.rectangle(img1, (row[1],row[2]),(row[1]+row[4],row[2]+row[3]),(0,0,255),1)
cv2.imshow("Display", img1)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite("result.jpg", img1)
