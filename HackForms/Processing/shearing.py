import cv2
import numpy as np
import imutils
def shear():
    image = cv2.imread("0001(1).jpg")
    image = imutils.resize(image, width=1000)
    cv2.imshow("original",image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)
    angle=0.5
    M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
    # M[0, 2] += (nW / 2)
    # M[1, 2] += (nH / 2)

    # perform the actual rotation and return the image
    image = cv2.warpAffine(image, M, (nW, nH))
    cv2.imshow("sheared", image)
    cv2.waitKey(0)
    image = imutils.resize(image, width=500)
    cv2.imshow("sheared2", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite("shear.jpg",image)
shear()