import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from math import ceil

def quick_show(img):
    cv.imshow('monit', img)
    cv.waitKey()
    cv.destroyAllWindows()


def load_images():
    template = cv.imread("template.jpg")

    hsv_image = hsv_image = cv.cvtColor(template, cv.COLOR_BGR2HSV)
    low_red = np.array([0, 100, 100])
    up_red = np.array([10, 255, 255])
    mask1 = cv.inRange(hsv_image, low_red, up_red)
    low_red = np.array([170, 50, 50])
    up_red = np.array([180, 255, 255])
    mask2 = cv.inRange(hsv_image, low_red, up_red)
    mask = mask1 + mask2
    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    BGRimage = cv.cvtColor(mask, cv.COLOR_GRAY2BGR)
    contours = sorted(contours, key=lambda x: cv.contourArea(x), reverse=True)
    contour = contours[0]
    cv.drawContours(BGRimage, contour, -1, (0, 0, 255), 2)

    # определяем границы тел
    c = cv.approxPolyDP(contour, 3, True)
    boundRect = cv.boundingRect(c)
    with_box = BGRimage
    x , y, w, h = boundRect 
    cv.rectangle(with_box, (x, y), (x + w, y + h), (255, 0, 0), 5)
    cutted = mask[y:y + h, x:x + w]
    #cv.imshow("template", cutted)
    #cv.waitKey()
    #cv.destroyAllWindows()
    return cutted


# Mask by red
img = cv.imread('./images/IMG_20240823_170049.jpg')
img2 = img
H, W, _ = img.shape
hsv_image = cv.cvtColor(img, cv.COLOR_BGR2HSV)
low_red = np.array([0, 100, 100])
up_red = np.array([10, 255, 255])
mask1 = cv.inRange(hsv_image, low_red, up_red)
low_red = np.array([170, 50, 50])
up_red = np.array([180, 255, 255])
mask2 = cv.inRange(hsv_image, low_red, up_red)
mask = mask1 + mask2
#cv.imshow("red_mask", mask)


template = load_images()
#quick_show(template)

# ищу контуры, обвожу, нахожу центер. На некоторых изо попадает в кадр пятый элемент( :)) ), добавил сортировку, взял 4 самых больших контура
contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
BGRimage = cv.cvtColor(mask, cv.COLOR_GRAY2BGR)
contours = sorted(contours, key=lambda x: cv.contourArea(x), reverse=True)
contours = contours[:2]
cv.drawContours(BGRimage, contours, -1, (0, 0, 255), 2)
# cv.imshow('contours', BGRimage)

# определяем границы тел
bodys = []
for contour in contours:
    c = cv.approxPolyDP(contour, 3, True)
    boundRect = cv.boundingRect(c)
    with_box = BGRimage
    x , y, w, h = boundRect 
    cv.rectangle(with_box, (x, y), (x + w, y + h), (255, 0, 0), 5)
    cutted = mask[y:y + h, x:x + w]
    bodys.append((cutted, (x, y, w, h)))
w, h = template.shape[::-1]

# масштабируем и сравниваем с шаблоном
results = []
for i in bodys:
    img = i[0]
    img = cv.resize(img, (w+2, h+2))
    results = []
    method = getattr(cv, 'TM_CCOEFF_NORMED')
    res = cv.matchTemplate(img,template,method)
    min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
    #print(max_val)
    results.append((i[1], max_val))

# выбираем максимум, определяем положение
body = max(results, key=lambda x: x[1])
x, y, w, h = body[0]
center = x + w // 2
ans = ceil(center / (W // 4))
print(ans)
