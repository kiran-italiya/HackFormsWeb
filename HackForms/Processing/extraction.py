import numpy as np
import cv2
import pytesseract
import csv
import pandas as pd
import imutils
import copy


def contour(image):
	img=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
	_, threshold = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY)
	contours,_=cv2.findContours(threshold,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
	return contours
def findCircle(cnts,img1):
	if len(cnts) == 0:
		print("No contour found!!")
	else:
		print(len(cnts))
		print("Starting circle search")
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
#                        cv2.circle(img, center, 5, (0, 0, 255), -1)
		cv2.imshow("contour", img)
		cv2.waitKey(0)
		cv2.destroyAllWindows()
		cv2.imwrite("circle.png",img)
	return circles
def detect_rectangles(image):

	img=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
	_, threshold = cv2.threshold(img, 240, 255, cv2.THRESH_BINARY)
	contours,_=cv2.findContours(threshold,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
	coordinate=[]
	for cnt in contours:

		approx = cv2.approxPolyDP(cnt, 0.01*cv2.arcLength(cnt, True), True)
		if len(approx) == 4 and cv2.contourArea(cnt)>100:
			'''if 2000+co > cv2.contourArea(cnt) > co:         
				continue
				co=cv2.contourArea(cnt)'''# try changing the value in place of 2000 to get outer rectangles
			coordinate.append([approx[0][0][0],approx[0][0][1],approx[2][0][0],approx[2][0][1]])
			print(approx)

	print(coordinate[0])# top-left coordinate of 1 rectangle
	print(coordinate[1])# bottom-right coordinate of 1 rectangle
	print(coordinate[2])# top-left coordinate of 2 rectangle
	print(coordinate[3])# bottom-right coordinate of 2 rectangle
#    #and so on
#    for i in range(len(coordinate)):
#        cv2.circle(img,(coordinate[0],coordinate[1]),5,(0,255,0),5)
#        cv2.circle(img,(coordinate[2],coordinate[3]),5,(0,255,0),5)
	return img,coordinate



#
#df = pd.read_csv('data22.csv')
#print(df)
#print(df)
img = cv2.imread("imgcrop1.jpg")
img = imutils.resize(img,width = 1000)
_, threshold = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY_INV)
cnt = contour(img)
circles = findCircle(cnt, img)
l = len(circles)/5
print(len(circles))
diff = 30
sum = []
i = 1
coordinates = []
for row in circles:        
	x,y,r = row[0],row[1],row[2]
	arr = img[y:y+r,x:x+r]
	sum.append(np.sum(arr))
	cv2.imshow("crop",img[y:y+r,x:x+r])
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	i+=1

rectangle_image,rec_coordinate=detect_rectangles(img)
print(rec_coordinate)
for x1,y1,x2,y2 in rec_coordinate:
	coordinates.append([x1,y1,x2,y2])
df_box = pd.DataFrame(coordinates, columns = ['X1','Y1','X2','Y2'])

df_box = df_box.sort_values(by= ['Y1','X1']).reset_index(drop=True)
print("DataFrame of Rectangle",df_box)
x1,y1,x2,y2 = 0,0,0,0
for row in df_box.itertuples():
	index_v = row[0]
	X1,Y1,X2,Y2 = copy.deepcopy(row[1]),copy.deepcopy(row[2]),copy.deepcopy(row[3]),copy.deepcopy(row[4])
	if row[1]<10 and row[2]<10:
		df_box = df_box.drop(row[0])
	if X2<X1:
		df_box.loc[row[0]][0] = X2
		df_box.loc[row[0]][2] = X1
	if Y2<Y1:
		df_box.loc[row[0]][1] = Y2
		df_box.loc[row[0]][3] = Y1
	try:
		while(index_v in df_box.index and abs(y1-df_box.loc[index_v][1])<10):
			if index_v in df_box.index:
#                    print(row[0]-1, y1,index_v, df.loc[index_v][1])
				if abs(x1-df_box.loc[index_v][0])<diff and abs(y1-df_box.loc[index_v][1])<diff and abs(x2-df_box.loc[index_v][2])<diff and abs(y2-df_box.loc[index_v][3])<diff:
					df_box = df_box.drop(index_v)
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
sum = []
for row in df_box.itertuples():
	x1,y1,x2,y2 = row[1],row[2],row[3],row[4]
	h = y2-y1
	w = x2-x1
	print(x1,y1,x2,y2)
	arr = threshold[y1+int(0.2*h):y2-int(0.2*h),x1+int(0.2*w):x2-int(0.2*w)]
	sum.append(np.sum(arr))
	cv2.imshow("crop",arr)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	cv2.rectangle(img, (row[1],row[2]),(row[3],row[4]),(0,0,255),1)

print(df_box)
print(sum)
cv2.imshow("crop",threshold)
cv2.waitKey(0)
cv2.destroyAllWindows()