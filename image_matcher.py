import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

img_rgb = cv.imread('captured_image.jpg')
assert img_rgb is not None, "file could not be read, check with os.path.exists()"

img_gray = cv.cvtColor(img_rgb, cv.COLOR_BGR2GRAY)
template = cv.imread('cropped_image.jpg', cv.IMREAD_GRAYSCALE)
assert template is not None, "file could not be read, check with os.path.exists()"

w, h = template.shape[::-1]
res = cv.matchTemplate(img_gray,template,cv.TM_CCOEFF_NORMED)
threshold = 0.8
loc = np.where( res >= threshold)

count = 0 
for pt in zip(*loc[::-1]):
    cv.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0,255,0), 2)
    count += 1

img_rbg = cv.putText(img_rgb, "Total Match: " + str(count), (50,50), cv.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv.LINE_AA)

while True:
    cv.imwrite('detected_image.jpg',img_rgb)
    cv.imshow('Detected',img_rgb)
    if cv.waitKey(1) == ord('q'): # ESC
        break
cv.destroyAllWindows()