import cv2 as cv
 
img = cv.imread("./images/PS8Fx3-dsz-jBI-hISTaj.jpg")
cv.imshow("orig", img)
img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow("gray", img_gray)
w, h = img_gray.shape
count = 0
all_count = w * h
br = 200
for i in range(w):
    for j in range(h):
        if img_gray[i][j] >= br:
            count += 1
print(count / all_count * 100)

cv.waitKey()
cv.destroyAllWindows()
