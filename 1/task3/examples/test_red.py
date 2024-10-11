import cv2 as cv 
import numpy as np

img = cv.imread('./images/IMG_20240823_170049.jpg')
hsv_image = cv.cvtColor(img, cv.COLOR_BGR2HSV)

low_red = np.array([0, 100, 100])
up_red = np.array([10, 255, 255])
mask1 = cv.inRange(hsv_image, low_red, up_red)
cv.imshow("mask", mask1)
low_red = np.array([170, 50, 50])
up_red = np.array([180, 255, 255])
mask2 = cv.inRange(hsv_image, low_red, up_red)
cv.imshow("mask2", mask2)
cv.waitKey()
cv.destroyAllWindows()
result = mask1 + mask2
cv.imshow("red_mask", result)
cv.waitKey()
cv.destroyAllWindows()
