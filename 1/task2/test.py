import cv2 as cv
import numpy as np

# функция ограничения диапазона (8 бит)
def f(x: int) -> int:
    if x > 255:
        return 255
    elif x < 0:
        return 0
    else:
        return x

# импорт изображения
img = cv.imread("./images/IMG_20240823_174144.jpg")
img = cv.resize(img, None ,fx=0.5, fy=0.5)

# маска на широкий диапазон красного
hsv_image = cv.cvtColor(img, cv.COLOR_BGR2HSV)
lower_red = np.array([0, 100, 100])
upper_red = np.array([10, 255, 255])
mask = cv.inRange(hsv_image, lower_red, upper_red)
# cv.imshow("mask_red", mask)

# поиск контура куба
contours, hierarchy = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
contours = sorted(contours, key=lambda x: cv.contourArea(x), reverse=True)
contour = contours[0]
BGRimage = cv.cvtColor(mask, cv.COLOR_GRAY2BGR)
cv.drawContours(BGRimage, contour, -1, (0, 0, 255), 5)
# cv.imshow('cub_contour', BGRimage)

# ограничеваем куб в прямоугольник
c = cv.approxPolyDP(contour, 3, True)
boundRect = cv.boundingRect(c)
with_box = BGRimage
cv.rectangle(with_box, (boundRect[0], boundRect[1]), (boundRect[0] + boundRect[2], boundRect[1] + boundRect[3]), (255, 0, 0), 5)
# cv.imshow("rectangle", with_box)

# из центра прямоугольника находим нужный красный
center = (boundRect[0] + boundRect[2] // 2, boundRect[1] + boundRect[3] // 2)
with_center = BGRimage
cv.circle(with_center, center, 2, (0, 255, 0), 3)
#cv.imshow("center", with_center)
color = hsv_image[center[1]][center[0]]

# сужаем диапазон красного, накладываем новую маску
lower_red = np.array([f(color[0] - 1), f(color[1] - 10), f(color[2] - 10)])
upper_red = np.array([f(color[0] + 1), f(color[1] + 10), f(color[2] + 25)])
mask2 = cv.inRange(hsv_image, lower_red, upper_red)
# cv.imshow("mask_red2", mask2)

# находим новый контур, определяем размеры
contours, hierarchy = cv.findContours(mask2, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
contours = sorted(contours, key=lambda x: cv.contourArea(x), reverse=True)
contour = contours[0]
BGRimage = cv.cvtColor(mask2, cv.COLOR_GRAY2BGR)
cv.drawContours(BGRimage, contour, -1, (0, 0, 255), 5)
# cv.imshow('cub_contour', BGRimage)
size = round(cv.arcLength(contour, True) / 4, 2)
# print("side:",size, "diametr:", (2 * size ** 2) ** 0.5)

# отделяем тела от фона, убираем куб
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
_, mask_white = cv.threshold(gray, 120, 255, cv.THRESH_BINARY_INV)
cv.imshow("mask_without_white", mask_white)
mask_inv = cv.bitwise_not(mask)
img_1 = cv.bitwise_and(img, img, mask = mask_inv)
img_2 = cv.bitwise_and(img_1, img_1, mask = mask_white)
cv.imshow("object", img_2)

# убираем шумы
kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (8, 8))
img_2 = cv.cvtColor(img_2, cv.COLOR_BGR2GRAY)
without_noize = img_2
#= cv.morphologyEx(img_2, cv.MORPH_OPEN, kernel)
# cv.imshow("clear", without_noize)

# выделяем контур объекта
smth = cv.cvtColor(without_noize, cv.COLOR_GRAY2BGR)
contours2, hierarchy = cv.findContours(without_noize, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
contours2 = sorted(contours2, key=lambda x: cv.contourArea(x), reverse=True)
contour = contours2[0]
cv.drawContours(smth, contour, -1, (0, 0, 255), 5)
cv.imshow('contour', smth)

# находим ограничивающий прямоугольник, считаем минимальный размер захвата
c = cv.approxPolyDP(contour, 3, True)
boundRect = cv.boundingRect(c)
with_box = smth
cv.rectangle(with_box, (boundRect[0], boundRect[1]), (boundRect[0] + boundRect[2], boundRect[1] + boundRect[3]), (255, 0, 0), 5)
# cv.imshow("object_with_box", with_box)
w, h = boundRect[2], boundRect[3]
s = round((w ** 2 + h ** 2) ** 0.5, 2)

# вывод
print("yes") if s <= (2 * size ** 2) ** 0.5 else print("no")

cv.waitKey()
cv.destroyAllWindows()
