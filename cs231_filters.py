import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage import exposure

def Sharp_img(img):
	kernel3 = np.array([[-1, -1,  -1],
                   [-1,  9, -1],
                    [-1, -1,  -1]])
	res = cv2.filter2D(src=img, ddepth=-1, kernel=kernel3)

	return res

def Gray_img(img):
	res = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	return res


def Bw_img(img, t ):
	img_gray_sharp = cv2.cvtColor(Sharp_img(img), cv2.COLOR_BGR2GRAY)

	max_value = np.max( img_gray_sharp )
	t = t * max_value / 255
	#t = np.average(img_gray_sharp) - 25
	thresh, res = cv2.threshold(img_gray_sharp,t,255,cv2.THRESH_BINARY)

	return res

def Equal_img(img, grid = None):
	b, g, r = cv2.split(img)

	if grid is not None:
		clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(grid, grid))
	else:
		clahe = cv2.createCLAHE(clipLimit=3.0)
	clahe_b = clahe.apply(b)
	clahe_g = clahe.apply(g)
	clahe_r = clahe.apply(r) 
	
	res = cv2.merge((clahe_b, clahe_g, clahe_r))

	return res

def Gray_equal_img(img):
	img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
	res = clahe.apply(img_gray)

	return res

def Contrast_stretching_img(img ):
	p2, p98 = np.percentile(img, (2, 98))
	res = exposure.rescale_intensity(img, in_range=(p2, p98))

	return res

def Gray_Contrast_stretching_img(img ):
	img = cv2.cvtColor(Sharp_img(img), cv2.COLOR_BGR2GRAY)

	p2, p98 = np.percentile(img, (2, 98))
	res = exposure.rescale_intensity(img, in_range=(p2, p98))

	return res


######################################################
################# TESTING
'''	
List_images = []
# Original Image
_img = cv2.imread("IMG_5530.jpg")
List_images.append(_img)

# Origin -> Sharpening Image
img_sharp = Sharp_img(_img)
List_images.append(img_sharp)

# Origin -> Grayscale Image
img_gray = Gray_img(_img)
List_images.append(img_gray)

# Sharp -> B&W Image
img_bw = Bw_img(_img)
List_images.append(img_bw)

# Origin -> Equalization Image
img_equal = Equal_img(_img)
List_images.append(img_equal)

# Origin -> Equalizing Grayscale Image
img_equal_gray = Gray_equal_img(_img)
List_images.append(img_equal_gray)

# Origin -> Contrast Stretching Image
img_rescale = Contrast_stretching_img(_img)
List_images.append(img_rescale)

# Sharp -> Contrast Stretching Image
img_rescale = Contrast_stretching_img(_img,'sharp')
List_images.append(img_rescale)

# Original -> Contrast Stretching Grayscale Image
img_rescale = Contrast_stretching_img(_img,'gray')
List_images.append(img_rescale)

# Sharp -> Contrast Stretching Grayscale Image
img_rescale = Contrast_stretching_img(_img,'graysharp')
List_images.append(img_rescale)

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
'''
