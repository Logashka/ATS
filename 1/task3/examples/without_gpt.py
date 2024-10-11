import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from math import ceil

class Body():
    def __init__(self, image, x: int):
        self.img = image
        self.pos = x

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
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    _, mask = cv.threshold(gray, 90, 255, cv.THRESH_BINARY_INV)
    # cv.imshow("mask", mask)

    # РёС‰Сѓ РєРѕРЅС‚СѓСЂС‹, РѕР±РІРѕР¶Сѓ, РЅР°С…РѕР¶Сѓ С†РµРЅС‚РµСЂ. РќР° РЅРµРєРѕС‚РѕСЂС‹С… РёР·Рѕ РїРѕРїР°РґР°РµС‚ РІ РєР°РґСЂ РїСЏС‚С‹Р№ СЌР»РµРјРµРЅС‚( :)) ), РґРѕР±Р°РІРёР» СЃРѕСЂС‚РёСЂРѕРІРєСѓ, РІР·СЏР» 4 СЃР°РјС‹С… Р±РѕР»СЊС€РёС… РєРѕРЅС‚СѓСЂР°
    contours, hierarchy = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    BGRimage = cv.cvtColor(mask, cv.COLOR_GRAY2BGR)
    contours = sorted(contours, key=lambda x: cv.contourArea(x), reverse=True)
    contours = contours[:4]
    cv.drawContours(BGRimage, contours, -1, (0, 0, 255), 2)
    # cv.imshow('contours', BGRimage)

    #С„РѕСЂРјРёСЂСѓСЋ РјР°СЃСЃРёРІ СЃ РѕР±РѕСЂРµР·Р°РЅРЅС‹РјРё РёР·РѕР±СЂР°Р¶РµРЅРёСЏРјРё
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

    # СЃРѕСЂС‚РёСЂСѓРµРј РѕР±СЂРµР·РєРё РёР·РѕР±СЂР°Р¶РµРЅРёСЏ СЃР»РµРІР° РЅР°РїСЂР°РІРѕ 
    bodys = sorted(bodys, key=lambda x: x.pos)

    # РїСЂРѕРІРµСЂСЏРµРј РЅР° СЃРѕРѕС‚РІРµС‚СЃРІРёРµ
    for b in bodys:
        img = b.img
        #cv.imshow("mon", img)
        #cv.waitKey()
        hsv_image = cv.cvtColor(img, cv.COLOR_BGR2HSV)
        x, y, _ = img.shape

        # РјР°СЃРєР° РЅР° РєСЂР°СЃРЅРѕРµ
        low_red = np.array([0, 100, 100])
        up_red = np.array([10, 255, 255])
        mask1 = cv.inRange(hsv_image, low_red, up_red)
        low_red = np.array([170, 50, 50])
        up_red = np.array([180, 255, 255])
        mask2 = cv.inRange(hsv_image, low_red, up_red)
        mask = mask1 + mask2
        #cv.imshow("mask", mask)
        #cv.waitKey()
        #cv.destroyAllWindows()

        # РµСЃР»Рё РЅРµ РєСЂР°СЃРЅРѕРµ - СЃРєРёРї
        if mask[x//2][y//2] == 0:
            continue
        
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        _, mask = cv.threshold(gray, 110, 255, cv.THRESH_BINARY_INV)
        contours, hierarchy = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        BGRimage = cv.cvtColor(mask, cv.COLOR_GRAY2BGR)
        contours = sorted(contours, key=lambda x: cv.contourArea(x), reverse=True)
        main, towers = contours[0], contours[1:4]

        approx = cv.approxPolyDP(main, 0.01 * cv.arcLength(main, True), True)
        if len(approx) < 10:
            continue
        
        points = []
        for i in towers:
            c = cv.approxPolyDP(i, 3, True)
            boundRect = cv.boundingRect(c)
            with_box = BGRimage
            x , y, w, h = boundRect 
            cv.rectangle(with_box, (x, y), (x + w, y + h), (255, 0, 0), 5)
            points.append((x + w // 2, y + h // 2))
        
        if is_it_true(points):
            ans = bodys.index(b) + 1
            return ans