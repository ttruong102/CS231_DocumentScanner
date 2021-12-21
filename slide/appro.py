import numpy as np
import cv2

path = "testContour.jpg"

img = cv2.imread( path )
img_gray = cv2.imread( path, 0 )

contour, _ = cv2.findContours( img_gray, cv2.RETR_EXTERNAL , cv2.CHAIN_APPROX_SIMPLE )

cv2.drawContours( img, contour, -1, (0,255,0), 1)

cv2.imshow( "asdfasdf",img )
cv2.waitKey(0)