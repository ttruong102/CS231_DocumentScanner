import cv2
import numpy as np


def combine2Images( im1, im2, axis = 1):
	return np.concatenate( (im1,im2), axis=axis)

def combineMultiImages( img_arr, row = 3, axis = 1 ):
	if ( len(img_arr) == 0): return
	# convert gray to rgb
	for i in range(len( img_arr ) ):
		if len(img_arr[i].shape) == 2:
			img_arr[i] = cv2.cvtColor( img_arr[i], cv2.COLOR_GRAY2RGB )

	rows = []
	row_count = 1
	rowImg = img_arr[0]
	i=1
	while i < len( img_arr ):
		if i % 3 != 0:
			rowImg = combine2Images( rowImg, img_arr[i] )
			row_count += 1
		else:
			rowImg = img_arr[i].copy()
			row_count = 1

		if row_count == 3 :
			rows.append(rowImg)

		i += 1
		
	need = 3-len( img_arr ) % 3
	if need != 3:
		h,w,d = img_arr[0].shape
		needImg = np.zeros( (h,w*need,d), dtype=np.uint8)
		rowImg = combine2Images(rowImg, needImg)
		rows.append( rowImg)


	res = rows[0]
	for i in range(1, len(rows ) ):
		res = combine2Images( res, rows[i] , axis=0)

	return res 

path = "result.jpg"
path = "testDilate.jpg"
path = "Edgecrop.jpg"
img = cv2.imread( path )


kernel = np.ones((3,3), np.uint8)

img_dilate = cv2.dilate( img, kernel )
img_erode = cv2.erode( img_dilate, kernel )

img_comb = combineMultiImages( [img, img_dilate, img_erode ])

cv2.imshow( "asdf", img_comb )
cv2.waitKey(0)

cv2.imwrite( "Connect.jpg", img_comb )




