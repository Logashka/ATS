import cv2
import numpy as np

def predict_illumination(image) -> bool:
    img_grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    w, h = img_grey.shape
    count = 0
    all_count = w * h
    br = 200
    for i in range(w):
        for j in range(h):
            if img_grey[i][j] >= br:
                count += 1
    percent = round(count / all_count * 100)
    return percent < 20