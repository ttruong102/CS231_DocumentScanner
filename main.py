import cs231
from cs231_filters import *
import numpy as np
import cv2

path = "test.jpg"
path = "img/1.jpg"
path = "img/IMG_1128.JPG"
path = "img/IMG_1135.JPG"
#nhan Q de ket thuc ham cs321.run()
process, result = cs231.run_by_path( path )
cv2.destroyAllWindows()


def nothing(x):
	pass

filters_names = ["sharp","gray","B&W","Equal","Gray Equal","Contrast stretching", "Gray contrast stretching"]
filters = [Sharp_img, Gray_img, Bw_img, Equal_img, Gray_equal_img, Contrast_stretching_img, Gray_Contrast_stretching_img]
cv2.namedWindow("filters control")
cv2.resizeWindow("filters control", 360, 240)
cv2.createTrackbar( "Filters","filters control", 0,len(filters_names)-1, nothing)
cv2.createTrackbar( "Thresh for B&W","filters control", 128,255, nothing)


if result.size == 0:
	print( "NO RESULT" )
else:
	while 1 :
		filter_index = cv2.getTrackbarPos( "Filters", "filters control" )
		filter_thresh = cv2.getTrackbarPos( "Thresh for B&W", "filters control")
		origin = result.copy()
		if filters_names[filter_index] != "B&W":
			filtered = filters[filter_index]( result )
		else:
			filtered = Bw_img( result, filter_thresh)

		filtered_for_show = filtered.copy()

		# font
		font = cv2.FONT_HERSHEY_SIMPLEX
		
		# org
		org = (5, 20)

		# fontScale
		fontScale = 0.8

		# Blue color in BGR
		color = (0, 0, 0)

		# Line thickness of 2 px
		thickness = 2

		# Using cv2.putText() method
		origin_for_show = cv2.putText(origin, "origin", org, font, 
		                   fontScale, color, thickness, cv2.LINE_AA)	
		
		filtered_for_show = cv2.putText(filtered_for_show, filters_names[filter_index], org, font, 
		                   fontScale, color, thickness, cv2.LINE_AA)	

		for_show = cs231.combine2Images( origin_for_show, filtered_for_show )

		cv2.imshow( "filter", for_show)

		if cv2.waitKey(1) == ord("q"):
			break
	
	cv2.destroyAllWindows()

	cv2.imwrite( "scan.jpg", result )
	cv2.imwrite( "filter.jpg", filtered )


	

