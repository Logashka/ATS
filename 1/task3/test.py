import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from math import ceil, atan2, degrees

class Body():
    def __init__(self, image, x: int):
        self.img = image
        self.pos = x

# Функция для нахождения самой выступающей точки на контуре
def find_farthest_point(contour, center):
    max_dist = 0
    farthest_point = None
    for point in contour:
        point = point[0]  # Убираем лишний уровень вложенности
        dist = np.linalg.norm(point - center)  # Вычисляем расстояние до центра
        if dist > max_dist:
            max_dist = dist
            farthest_point = tuple(point)
    return farthest_point


# Функция для вычисления угла между двумя линиями
def angle_between_lines(p1, p2, p3, p4):
    line1 = np.array([p2[0] - p1[0], p2[1] - p1[1]])
    line2 = np.array([p4[0] - p3[0], p4[1] - p3[1]])
    
    dot_product = np.dot(line1, line2)
    mag1 = np.linalg.norm(line1)
    mag2 = np.linalg.norm(line2)
    
    cos_theta = dot_product / (mag1 * mag2)
    angle = degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0)))
    return angle



# Функция для проверки, является ли треугольник равносторонним (с допуском 5%)
def is_equilateral(centers, tolerance=0.05):
    # Вычисляем длины сторон треугольника
    d1 = np.linalg.norm(np.array(centers[0]) - np.array(centers[1]))
    d2 = np.linalg.norm(np.array(centers[1]) - np.array(centers[2]))
    d3 = np.linalg.norm(np.array(centers[0]) - np.array(centers[2]))
    
    avg = (d1 + d2 + d3) / 3
    return all(abs(d - avg) <= avg * tolerance for d in [d1, d2, d3])


def is_it_true(points: tuple, br = 0.25)->bool:
    first = ((points[0][0] - points[1][0]) ** 2 + (points[0][1] - points[1][1]) ** 2) ** 0.5
    second = ((points[0][0] - points[2][0]) ** 2 + (points[0][1] - points[2][1]) ** 2) ** 0.5
    third = ((points[1][0] - points[2][0]) ** 2 + (points[1][1] - points[2][1]) ** 2) ** 0.5
    k1 = first / second
    k2 = first / third 
    k3 = second / third
    if 1 - br <= k1 <= 1 + br and 1 - br <= k2 <= 1 + br and 1 - br <= k3 <= 1 + br:
        return True
    else: 
        return False

def load_images():
    return []

def predict_connect_number(img, img_list) -> int:
    # накладываем маску по серому
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    _, mask = cv.threshold(gray, 90, 255, cv.THRESH_BINARY_INV)
    # cv.imshow("mask", mask)

    # ищем контуры
    contours, hierarchy = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    BGRimage = cv.cvtColor(mask, cv.COLOR_GRAY2BGR)
    contours = sorted(contours, key=lambda x: cv.contourArea(x), reverse=True)
    contours = contours[:4]
    cv.drawContours(BGRimage, contours, -1, (0, 0, 255), 2)
    cv.imshow('contours', BGRimage)

    # нарезаем изображения 
    bodys = []
    for contour in contours:
        c = cv.approxPolyDP(contour, 3, True)
        boundRect = cv.boundingRect(c)
        with_box = BGRimage
        x , y, w, h = boundRect 
        cv.rectangle(with_box, (x, y), (x + w, y + h), (255, 0, 0), 5)
        cutted = img[y:y + h, x:x + w]
        bodys.append(Body(cutted, x))
        #cv.imshow("mon", cutted)
        #cv.waitKey()

    # сортируем слева направо
    bodys = sorted(bodys, key=lambda x: x.pos)
    print(len(bodys))

    for b in bodys:
        img = b.img
        #cv.imshow("mon", img)
        #cv.waitKey()
        hsv_image = cv.cvtColor(img, cv.COLOR_BGR2HSV)
        x, y, _ = img.shape

        # маска на красное 
        low_red = np.array([0, 100, 100])
        up_red = np.array([10, 255, 255])
        mask1 = cv.inRange(hsv_image, low_red, up_red)
        low_red = np.array([170, 50, 50])
        up_red = np.array([180, 255, 255])
        mask2 = cv.inRange(hsv_image, low_red, up_red)
        mask = mask1 + mask2
        # cv.imshow("mask", mask)
        # cv.waitKey()
        # cv.destroyAllWindows()

        # проверка на красное
        if mask[x//2][y//2] == 0:
            continue
        
        # выделяем контуры 
        #gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        #_, mask = cv.threshold(gray, 110, 255, cv.THRESH_BINARY_INV)
        contours, hierarchy = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        BGRimage = cv.cvtColor(mask, cv.COLOR_GRAY2BGR)
        contours = sorted(contours, key=lambda x: cv.contourArea(x), reverse=True)
        main, towers = contours[0], contours[1:4]

        # проверка на круг
        approx = cv.approxPolyDP(main, 0.01 * cv.arcLength(main, True), True)
        if len(approx) < 10:
            continue
        
        # координаты середин белых башенок
        points = []
        for i in towers:
            c = cv.approxPolyDP(i, 3, True)
            boundRect = cv.boundingRect(c)
            with_box = BGRimage
            x , y, w, h = boundRect 
            cv.rectangle(with_box, (x, y), (x + w, y + h), (255, 0, 0), 5)
            points.append((x + w // 2, y + h // 2))
        
        # находим центральную точку контура
        M = cv.moments(main)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        center_red = np.array([cX, cY])
        
        # находим выступ
        farthest_point = find_farthest_point(main, center_red)

        # Найдем ближайший белый штырек к выступу
        distances_to_farthest = [np.linalg.norm(np.array(center) - np.array(farthest_point)) for center in points]
        closest_idx = np.argmin(distances_to_farthest)
        closest_pin = points[closest_idx]
        
        # Формируем две линии:
        # 1. Линия между ближайшим белым штырьком и выступом
        # 2. Линия между двумя оставшимися штырьками
        remaining_pins = [points[i] for i in range(3) if i != closest_idx]
        
        # Проверим угол между линиями
        if len(remaining_pins) == 2:
            angle = angle_between_lines(closest_pin, farthest_point, remaining_pins[0], remaining_pins[1])
            
            if abs(angle - 90) <= 10:
                ans = bodys.index(b) + 1
                #print("Линии перпендикулярны с точностью до 10 градусов.")
                return ans
            else:
                pass
                #print(f"Угол между линиями: {angle:.2f} градусов. Линии НЕ перпендикулярны.")
        else:
            pass
            #print("Не удалось найти два других белых штырька для проверки.")

image = cv.imread("images/IMG_20240823_172945.jpg")
user_answer = predict_connect_number(image, [])
print(user_answer)