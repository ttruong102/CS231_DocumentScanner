import cv2
import numpy as np
import utlis
import phu_math
import os 

DEFAULT_THRES1 = 70
DEFAULT_THRES2 = 70
DEFAULT_BLUR = 4
DEFAULT_THICK = 4

PROCESS_HEIGHT = 700
		


def resizeImagebyH( img, want_h=PROCESS_HEIGHT) :
	h,w = img.shape[0], img.shape[1]

	want_w = int( want_h * (w/h))
	return cv2.resize( img, (want_w,want_h))

def resizeImagebyW( img, want_w=1800 ) :
	h,w = img.shape[0], img.shape[1]

	want_h = int( want_w * (h/w))
	return cv2.resize( img, (want_w,want_h))

def combine2Images( im1, im2, axis = 1):
	if len(im2.shape) == 2:
		im2 = cv2.cvtColor( im2, cv2.COLOR_GRAY2RGB )
	if len(im1.shape) == 2:
		im1 = cv2.cvtColor( im1, cv2.COLOR_GRAY2RGB )
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

	res = resizeImagebyH( res)
	if res.shape[1] > 1800:
		res = resizeImagebyW( res)

	return res 


def warpByOriginImage( org, A,B,C,D ):

	h = org.shape[0]
	# add 2 upper
	# pointsNew is ABDC
	pointsNew = np.zeros((4, 1, 2), dtype=np.int32)
	pointsNew[0] = A* h/PROCESS_HEIGHT
	pointsNew[1] = B* h/PROCESS_HEIGHT
	pointsNew[2] = D* h/PROCESS_HEIGHT
	pointsNew[3] = C* h/PROCESS_HEIGHT


	# get ABCD for find ratio

	ratio = phu_math.calculateRatio( A,B,C,D )

	warpHeight = 700
	warpWidth = int (warpHeight * ratio)

	pointsSorted = np.float32( pointsNew )
	pointsDes = np.float32([[0, 0],[warpWidth, 0], [0, warpHeight],[warpWidth, warpHeight]]) # PREPARE POINTS FOR WARP

	matrix = cv2.getPerspectiveTransform( pointsSorted, pointsDes)

	imgPerTrans = cv2.warpPerspective( org , matrix, (warpWidth,warpHeight))

	return imgPerTrans, matrix


def nothing(x): pass

cv2.namedWindow("Trackbars")
cv2.resizeWindow("Trackbars", 360, 240)
cv2.createTrackbar("Threshold1", "Trackbars", DEFAULT_THRES1,255, nothing)
cv2.createTrackbar("Threshold2", "Trackbars", DEFAULT_THRES2, 255, nothing)
cv2.createTrackbar("Blur", "Trackbars", DEFAULT_BLUR, 20, nothing)
cv2.createTrackbar("Thick", "Trackbars", DEFAULT_THICK, 20, nothing)


def run_on_webcam( ):
	cap = cv2.VideoCapture(0)

	processingImage = np.array([])
	result = np.array([])

	while 1:
		ret, origin_img_color  = cap.read()

		if ret :
			processingImage, result = run_one_loop( origin_img_color )
		
		if cv2.waitKey(1) == ord('q'):
			break
	return processingImage, result

def run_by_path( path ):
	# do 
	origin_img_color = cv2.imread( path)

	processingImage = np.array([])
	result = np.array([])

	# resize 
	while 1:
		processingImage, result = run_one_loop( origin_img_color )

		if cv2.waitKey(1) == ord('q'):
			break
	
	return processingImage, result

def run_one_loop( origin_img_color ):
	src_img = cv2.cvtColor( origin_img_color, cv2.COLOR_BGR2GRAY )

	src_img_color = resizeImagebyH( origin_img_color)
	src_img = resizeImagebyH( src_img)


	contours_index = 0
	thick = 10

	img_color = src_img_color.copy()
	img = src_img.copy()
	#get trackbar pos
	thres1 = cv2.getTrackbarPos( "Threshold1","Trackbars")
	thres2 = cv2.getTrackbarPos( "Threshold2","Trackbars")
	thick = cv2.getTrackbarPos( "Thick","Trackbars")
	blur = cv2.getTrackbarPos( "Blur","Trackbars") 
	blur = blur * 2 + 1

	processingIMGs = []
	# canny edge detection

	img = cv2.GaussianBlur(img, (blur,blur), cv2.BORDER_DEFAULT)
	processingIMGs.append( img.copy() )
	img = cv2.Canny( img, thres1,thres2)
	processingIMGs.append( img.copy() )



	# dilate and erode
	kernel = np.ones((5,5), np.uint8)
	img = cv2.dilate(img, kernel)
	processingIMGs.append( img.copy() )

	img = cv2.erode(img, kernel)
	processingIMGs.append( img.copy() )


	contours, _ = cv2.findContours( img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE )
	cv2.drawContours( img_color, contours, -1 , (0,255,0), thick )
	processingIMGs.append( img_color.copy() )
	#cv2.imshow("test", img_color )
	#draw contour for power points
	img_Contours = np.zeros( src_img_color.shape, np.uint8 )
	img_Rect = np.zeros( src_img_color.shape, np.uint8 )
	img_Rect = src_img_color.copy()
	colors = [(255,0,0),(0,255,0),(0,0,255),(255,255,0),(255,0,255),(0,255,255)]
	for i in range(len(contours)):
		cv2.drawContours( img_Contours, [contours[i]], -1, colors[i%len(colors)], thick)

	# cv2.imshow("contours", img_Contours )

	imgPerTrans = np.array([])
	pointsNew = np.array([])
	if len( contours)> 0 :
		this_contour,_ = utlis.biggestContour(contours)
		if len(this_contour) > 0 :
			#cv2.drawContours( img_color, contours, -1 , (0,255,0), thick
			appro = cv2.approxPolyDP( this_contour, 0.01*cv2.arcLength( this_contour, True), True)
			img_color = src_img_color.copy()
			cv2.drawContours( img_color, [appro], -1 , (0,255,0), thick)
			cv2.drawContours( img_Rect, [appro], -1 , (0,255,0), thick)
			processingIMGs.append( img_color )


			points = np.reshape( appro, (4,2))

			avgY = 0
			for p in points:
				avgY += p[1]
			avgY /= 4

			upper = []
			lower = []
			#get 2 uppers, 2 lowers
			for p in points:
				if p[1] <= avgY:
					upper.append( p )
				else:
					lower.append( p )

			def sortByX( arr ):
				def swap(a,b):
					temp = a
					a = b 
					b = temp

					return a,b

				n = len(arr )
				for i in range(n):
					for j in range(i+1,n):
						if arr[j][0] < arr[i][0]:
							arr[j],arr[i] = swap( arr[j],arr[i] )

				return arr

			upper = sortByX(upper)
			lower = sortByX(lower)

			A,B,C,D = upper + lower[::-1]


			imgPerTrans, matrix = warpByOriginImage( origin_img_color, A,B,C,D)
			cv2.imshow( "warp", imgPerTrans ) 


	processImg = combineMultiImages( processingIMGs ) 
	cv2.imshow("processing", processImg)


	if contours_index >= len(contours):
		contours_index = len(contours) -1 
	if contours_index < 0:
		contours_index = 0 

	return processImg, imgPerTrans


