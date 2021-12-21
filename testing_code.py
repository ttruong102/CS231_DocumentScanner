import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage import exposure

List_images = []

# Original Image
_img = cv2.imread("test.jpg")

List_images.append(_img)

# Origin -> Sharpening Image
kernel3 = np.array([[0, -1,  0],
                   [-1,  5, -1],
                    [0, -1,  0]])
img_sharp = cv2.filter2D(src=_img, ddepth=-1, kernel=kernel3)

List_images.append(img_sharp)

# Origin -> Grayscale Image
img_gray = cv2.cvtColor(_img, cv2.COLOR_BGR2GRAY)

List_images.append(img_gray)

# Sharp -> B&W Image
img_gray_sharp = cv2.cvtColor(img_sharp, cv2.COLOR_BGR2GRAY)
t = np.average(img_gray_sharp) - 25
thresh, img_bw = cv2.threshold(img_gray_sharp,t,255,cv2.THRESH_BINARY)

List_images.append(img_bw)

# Origin -> Equalization Image
def Adapt_equal(img, grid = None):
  b, g, r = cv2.split(img)
  if grid is not None:
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(grid, grid))
  else:
    clahe = cv2.createCLAHE(clipLimit=3.0)
  clahe_b = clahe.apply(b)
  clahe_g = clahe.apply(g)
  clahe_r = clahe.apply(r) 
  equa_img = cv2.merge((clahe_b, clahe_g, clahe_r))
  return equa_img

img_equal = Adapt_equal(_img,8)

List_images.append(img_equal)

# Origin -> Equalizing Grayscale Image
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
img_equal_gray = clahe.apply(img_gray)

List_images.append(img_equal_gray)

#img_equal_gray_s = clahe.apply(img_gray_sharp)

# Origin -> Contrast Stretching Image
p2, p98 = np.percentile(_img, (2, 98))
img_rescale = exposure.rescale_intensity(_img, in_range=(p2, p98))

List_images.append(img_rescale)

# Sharp -> Contrast Stretching Image
p2, p98 = np.percentile(_img, (2, 98))
img_rescale_sharp = exposure.rescale_intensity(img_sharp, in_range=(p2, p98))

List_images.append(img_rescale_sharp)

# Original -> Contrast Stretching Grayscale Image
p2, p98 = np.percentile(img_gray, (2, 98))
img_rescale_gray = exposure.rescale_intensity(img_gray, in_range=(p2, p98))

List_images.append(img_rescale_gray)

# Sharp -> Contrast Stretching Grayscale Image
p2, p98 = np.percentile(img_gray_sharp, (2, 98))
img_rescale_graysharp = exposure.rescale_intensity(img_gray_sharp, in_range=(p2, p98))

List_images.append(img_rescale_graysharp)

""" for i in List_images:
	cv2.resizeWindow("output", 200, 300)     
	cv2.imshow("",i)
	cv2.waitKey(0) """

plt.figure(num="Image Fig" ,figsize=(100,100))
title = ["Origin","Sharp Image","Grayscale","B&W","Equalizing","Grayscale Equalizing"
		,"Contrast","Sharp Contrast","Grayscale Contrast"
		,"Grayscale Sharp Contrast"]
idx = 1
for i, t in zip(List_images,title):
	img = cv2.cvtColor(i, cv2.COLOR_BGR2RGB)
	plt.subplot(2, 5, idx)
	plt.title(t)
	plt.axis('off')
	plt.imshow(img,cmap='gray')
	idx+=1
plt.show()