import numpy as np
import math
#import matplotlib.pyplot as plt


def scaleABCD( A,B,C,D ):
	maxValue = 0
	for p in (A,B,C,D):
		if p[0] > maxValue: maxValue = p[0]
		if p[1] > maxValue: maxValue = p[1]
	
	A = ( A[0] / maxValue , A[1]/maxValue)
	B = ( B[0] / maxValue , B[1]/maxValue)
	C = ( C[0] / maxValue , C[1]/maxValue)
	D = ( D[0] / maxValue , D[1]/maxValue)

	return A,B,C,D


def GiaiHePT2An( a1,a2,a3, b1,b2,b3):
	# ham nay giai phuong trinh
	# a1X + a2Y = a3
	# b1X + b2Y = b3
	# tra ve (X,Y)

	#cong thuc Cramer
	x = ( a3*b2 -  b3*a2 ) / ( a1*b2 - b1*a2 )
	y = ( a1*b3 -  b1*a3 ) / ( a1*b2 - b1*a2 )

	return (x,y)

def giaiPTBac2(a, b, c):
	# kiểm tra các hệ số
	if (a == 0):
		if (b == 0): 
			return False 
		else:
			x = -c / b
			return  x,x
	
	# tính delta
	delta = b * b - 4 * a * c
	# tính nghiệm
	if (delta > 0):
		x1 = (float)((-b + math.sqrt(delta)) / (2 * a))
		x2 = (float)((-b - math.sqrt(delta)) / (2 * a))
		return x1, x2
	elif (delta == 0):
		x1 = (-b / (2 * a))
		return x1, x1
	else:
		return False 

def isParallelLines( A,B,C,D):
	Ax, Ay = A
	Bx, By = B
	Cx, Cy = C
	Dx, Dy = D

	vector_1 = [A[0] - B[0], A[1] - B[1]]
	vector_2 = [C[0] - D[0], C[1] - D[1]]

	unit_vector_1 = vector_1 / np.linalg.norm(vector_1)
	unit_vector_2 = vector_2 / np.linalg.norm(vector_2)
	dot_product = np.dot(unit_vector_1, unit_vector_2)
	angle = np.arccos(dot_product)

	easyCalculate = 0.14

	isParallel = False
	if angle < math.pi + easyCalculate and angle > math.pi - easyCalculate :
		isParallel = True
	if angle < 0 + easyCalculate and angle > 0 - easyCalculate :
		isParallel = True

	return isParallel


	isABhorizon = False
	isCDhorizon = False
	
	isABvertical = False
	isCDvertical = False

	if Ay == By : isABhorizon = True
	if Cy == Dy : isCDhorizon = True

	if Ax == Bx : isABvertical = True
	if Cx == Dx : isCDvertical = True
	# check parallel
	ABparaCD = False
	if ( (isABvertical and isCDvertical) or (isABhorizon and isCDhorizon)):
		ABparaCD = True
	elif ( ((isABvertical or isCDvertical) and not (isABvertical and isCDvertical)) ):
		ABparaCD = False
	elif ( ((isABhorizon or isCDhorizon) and not (isABhorizon and isCDhorizon)) ):
		ABparaCD = False
	else: 
		same = ((Ax - Bx) / ( Ay - By )) / ( (Cx - Dx) / (Cy-Dy)) 
		if same >= 0.9 and same <= 1.1:
			ABparaCD = True 

	return ABparaCD
def findInterSectionPoint( A,B,C,D ):
	# if AB // CD, then return None
	if isParallelLines(A,B,C,D) : return None
	# this function find Intersection Point of AB and CD
	Ax, Ay = A
	Bx, By = B
	Cx, Cy = C
	Dx, Dy = D

	Ey ,Ex = GiaiHePT2An( Ax - Bx, By-Ay, Ax*By - Ay*Bx, Cx-Dx, Dy-Cy,Cx*Dy-Cy*Dx)

	return Ex,Ey

	isABhorizon = False
	isCDhorizon = False
	
	isABvertical = False
	isCDvertical = False

	if Ay == By : isABhorizon = True
	if Cy == Dy : isCDhorizon = True

	if Ax == Bx : isABvertical = True
	if Cx == Dx : isCDvertical = True

	if isParallelLines(A,B,C,D):
		return None
	
	# find point
	Ex = 0
	Ey = 0

	if ( isABvertical and isCDhorizon ):
		Ex = Ax
		Ey = Cy
	elif ( isABhorizon and isCDvertical):
		Ex = Cx
		Ey = Ay
	elif ( isABvertical ):
		M2 = (Dx - Cx) / (Dy - Cy)
		Ex = Ax 
		Ey = - (Dx - Dy*M2 - Ex) / M2
	elif ( isCDvertical ):
		M1 = (Ax - Bx) / (Ay - By)
		Ex = Cx 
		Ey = - (Bx - By*M1 - Ex) / M1
	elif isABhorizon:
		M2 = (Dx - Cx) / (Dy - Cy)
		Ey = Ay
		Ex = Dx - Dy*M2 + Ey*M2
	elif isCDhorizon:
		M1 = (Ax - Bx) / (Ay - By)
		Ey = Cy
		Ex = Bx - By*M1 + Ey*M1
	else:
		M1 = (Ax - Bx) / (Ay - By)
		M2 = (Dx - Cx) / (Dy - Cy)
		return GiaiHePT2An( 1, -M1, Ax - Ay*M1, 1,-M2, Dx-Dy*M2)
	
	return (Ex,Ey)


def findHorizonLine( A,B,C,D ):
	Ax, Ay = A
	Bx, By = B
	Cx, Cy = C
	Dx, Dy = D

	parallelLinesCount = 0
	parallelLines = ""

	isABhorizon = False
	isADhorizon = False
	isCBhorizon = False
	isCDhorizon = False
	
	isABvertical = False
	isADvertical = False
	isCBvertical = False
	isCDvertical = False

	if Ay == By : isABhorizon = True
	if Ay == Dy : isADhorizon = True
	if By == Cy : isCBhorizon = True
	if Cy == Dy : isCDhorizon = True

	if Ax == Bx : isABvertical = True
	if Ax == Dx : isADvertical = True
	if Bx == Cx : isCBvertical = True
	if Cx == Dx : isCDvertical = True

	# AB // CD ?
	if isParallelLines(A,B,C,D):
		parallelLinesCount += 1
		parallelLines = "AB // CD"
		
	# AD // BC ?
	if isParallelLines( A,D,B,C):
		parallelLinesCount += 1
		parallelLines = "AD // BC"


	#main
	if parallelLinesCount == 2 :
		return (parallelLinesCount, True, True)


	if parallelLinesCount == 1:
		#find Angle
		point = (0,0)
		angle = 0
		if ( parallelLines == "AB // CD"):
			#find angle of AB
			if isABhorizon :
				angle = "horizon"
			elif isABvertical:
				angle = "vertical"
			else:
				angle = (By - Ay) / (Bx - Ax)

			# find intersection point of AD and BC
			point = findInterSectionPoint( A,D,C,B)
		else:
			#find angle of AD
			if isADhorizon :
				angle = "horizon"
			elif isADvertical:
				angle = "vertical"
			else:
				angle = (Dy - Ay) / (Dx - Ax)
			# find intersection point of AB and CD 
			point = findInterSectionPoint( A,B,C,D)


		return  (parallelLinesCount ,angle, point )
	
	if parallelLinesCount == 0 :
		# find 2 intersection points
		point1 = findInterSectionPoint( A,B,C,D)
		point2 = findInterSectionPoint( A,D,C,B)

		return (parallelLinesCount,point1, point2)

def old_calculateRatio( A,B,C,D ):

	A,B,C,D = scaleABCD( A,B,C,D )

	# return real AB / AD 
	horizoneLineInfo = findHorizonLine( A,B,C,D ) 
	parallelLinesCount = horizoneLineInfo[0]

	A = np.array(A)
	B = np.array(B)
	C = np.array(C)
	D = np.array(D)

	#if straight view 
	if parallelLinesCount == 2:
		# caculate length of AB and AD
		AB = np.linalg.norm( A-B )
		AD = np.linalg.norm( A-D )

		return AB/AD

	if parallelLinesCount == 1:
		deg = 0
		horLinePoint1 = horizoneLineInfo[2]
		horLinePoint2 = (0,0)

		if horizoneLineInfo[1] == "vertical":
			deg = 90
			horLinePoint2 = ( horLinePoint1[0], horLinePoint1[1]+1 )
		elif horizoneLineInfo[1] == "horizon":
			deg = 0
			horLinePoint2 = ( horLinePoint1[0]+1, horLinePoint1[1] )
		else:
			tan = horizoneLineInfo[1]
			horLinePoint2 = (horLinePoint1[0] + 1, horLinePoint1[1] + tan)
			deg = math.degrees( math.atan( tan ) )
		
		# rotate 
		theta = np.deg2rad(-deg)
		rot = np.array([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]])

		rot_A = np.array(A).copy()
		rot_B = np.array(B).copy()
		rot_C = np.array(C).copy()
		rot_D = np.array(D).copy()
		rot_horLinePoint2 = np.array(horLinePoint2).copy()
		rot_horLinePoint1 = np.array(horLinePoint1).copy()

		rot_A = np.dot(rot,rot_A)
		rot_B = np.dot(rot,rot_B)
		rot_C = np.dot(rot,rot_C)
		rot_D = np.dot(rot,rot_D)
		rot_horLinePoint1 = np.dot(rot,rot_horLinePoint1)
		rot_horLinePoint2 = np.dot(rot,rot_horLinePoint2)

		# calculate ratio
		# determine which edges couple is parallel
		isABparallel = False
		Edge1 = 0
		Edge2 = 0
		if isParallelLines( A,B,C,D ):
			isABparallel = True

			AB = np.linalg.norm( A-B )
			CD = np.linalg.norm( C-D )

			if AB > CD :
				Edge1 = AB
			else:
				Edge1 = CD 
		else :
			AD = np.linalg.norm( A-D )
			CB = np.linalg.norm( C-B )

			if AD > CB :
				Edge1 = AD
			else:
				Edge1 = CB 
		
		# cal real height
		yOfHor = rot_horLinePoint1[1]
		maxheight = abs(rot_A[1] - yOfHor)
		minheight = abs(rot_A[1] - yOfHor)
		for p in ( rot_A,rot_B,rot_C,rot_D):
			h = abs(p[1] - yOfHor)
			if h > maxheight : 
				maxheight = h
			if h < minheight:
				minheight = h

		height = maxheight - minheight

		Edge2 = (height*maxheight) / ( maxheight - height )
		if isABparallel :
			return Edge1 /Edge2
		return Edge2 / Edge1

	if parallelLinesCount == 0:
		horLinePoint1 = horizoneLineInfo[1]
		horLinePoint2 = horizoneLineInfo[2]

		deg = 0
		# if hor is vertical
		if horLinePoint1[0] == horLinePoint2[0]:
			deg = 90
		# if hor is hor
		elif horLinePoint1[1] == horLinePoint2[1]:
			deg = 0
		else:
			tan = (horLinePoint2[1] - horLinePoint1[1]) / ( horLinePoint2[0] - horLinePoint1[0] )
			deg = math.degrees( math.atan( tan ))


		#rotate 
		theta = np.deg2rad(-deg)
		rot = np.array([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]])

		rot_A = np.array(A).copy()
		rot_B = np.array(B).copy()
		rot_C = np.array(C).copy()
		rot_D = np.array(D).copy()
		rot_horLinePoint2 = np.array(horLinePoint2).copy()
		rot_horLinePoint1 = np.array(horLinePoint1).copy()

		rot_A = np.dot(rot,rot_A)
		rot_B = np.dot(rot,rot_B)
		rot_C = np.dot(rot,rot_C)
		rot_D = np.dot(rot,rot_D)
		rot_horLinePoint1 = np.dot(rot,rot_horLinePoint1)
		rot_horLinePoint2 = np.dot(rot,rot_horLinePoint2)

		yOfHor = rot_horLinePoint1[1]
		nearPoint = rot_A
		sidePoint1 = rot_A
		sidePoint2 = rot_A
		farPoint = rot_A

		maxheight = abs(rot_A[1] - yOfHor)
		minheight = abs(rot_A[1] - yOfHor)
		# find near, far point
		for p in ( rot_A,rot_B,rot_C,rot_D):
			h = abs(p[1] - yOfHor)
			if h > maxheight : 
				maxheight = h
				nearPoint = p
			if h < minheight:
				minheight = h
				farPoint = p

		#find side point
		index = 0
		for p in ( rot_A,rot_B,rot_C,rot_D):
			h = abs(p[1] - yOfHor)
			if h > minheight and h < maxheight:
				if index == 0:
					sidePoint1 = p
					index += 1
				else:
					sidePoint2 = p

		Side1IsA = False
		FarisB = False
		if sidePoint1[0] == rot_A[0] and sidePoint1[1] == rot_A[1]:
			Side1IsA = True
		if farPoint[0] == rot_B[0] and farPoint[1] == rot_B[1]:
			FarisB = True

		#get ratio from 2 far edge
		ratio = np.linalg.norm( sidePoint1 - farPoint ) / np.linalg.norm( sidePoint2 - farPoint )

		if ( ( Side1IsA and FarisB ) or ( not Side1IsA and not FarisB)):
			return ratio
		return 1/ratio

		## find intersection points of "far, sides" lines and near hor line
		#interPoint1 = findInterSectionPoint( farPoint, sidePoint1, nearPoint, (nearPoint[0]+1, nearPoint[1]))
		#interPoint2 = findInterSectionPoint( farPoint, sidePoint2, nearPoint, (nearPoint[0]+1, nearPoint[1]))

		## start calculate
		#h1 = abs( sidePoint1[1] - nearPoint[1])
		#h2 = abs( sidePoint2[1] - nearPoint[1])
		#h1 = (h1 * maxheight) / (maxheight - h1 )
		#h2 = (h2 * maxheight) / (maxheight - h2 )

		#interDis1 = np.linalg.norm( np.array(interPoint1) - np.array(nearPoint))
		#interDis2 = np.linalg.norm( np.array(interPoint2) - np.array(nearPoint))

		#c = interDis1
		#h = h1 
		#afterFix = h1/h2
		#if interDis2 < interDis1:
		#	c = interDis2
		#	h = h2
		#	afterFix = h2/h1

		#nghiem = giaiPTBac2( 1,-c*c, (c*h) *(c*h) )
		#a=0
		#b=0
		#if nghiem != False:
		#	b,a = nghiem 

		#a = math.sqrt(a)
		#b = math.sqrt(b)

		#ratio = (b/a) * afterFix

		#return ratio

def calculateRatio( A,B,C,D ):
	A,B,C,D = scaleABCD( A,B,C,D )
	A = np.array(A)
	B = np.array(B)
	C = np.array(C)
	D = np.array(D)

	#calculate length of 4 edges
	AB = np.linalg.norm( A-B )
	AD = np.linalg.norm( A-D )
	CB = np.linalg.norm( C-B )
	CD = np.linalg.norm( C-D )

	smallAB = AB
	smallAD = AD
	if AB > CD:
		smallAB = CD
	if AD > CB:
		smallAD = CB
	
	return smallAB/smallAD

		
#def plotLineFrom2Points( point1, point2, color = "blue" ):
	#x_values = [ point1[0], point2[0] ]
	#y_values = [ point1[1], point2[1] ]
	#plt.plot( x_values, y_values, color = color)
			
	
#### TEST ZONE ####
#A = (1,0)
#B = (0,2)
#C = (0,4)
#D = (2,0)
#
#A = (0,2)
#B = (6,0)
#C = (7,3)
#D = (1,5)
#
#A = (0,1)
#B = (2,0)
#C = (4,0)
#D = (0,2)
#
#A = (0,1)
#B = (2,0)
#C = (4,1)
#D = (3,4)
#
##rectangle ratio 2/1
#A = (274,579)
#B = (700,736)
#C = (789,612)
#D = (439,541)
#
##square
#A = (778,587)
#B = (506,474)
#C = (286,521)
#D = (452,804)
#
# drawing page
#A = ( 450,182 )
#B = ( 245,168 )
#C = (  19,373 )
#D = ( 267,487 )
#
#colors = ["red","blue","green","red"]
#for p,color in zip([A,B,C,D],colors):
#	plt.plot(p[0],p[1],"o", color=color)
#
#
#inter1 = findInterSectionPoint( A,B,C,D)
#inter2 = findInterSectionPoint( A,C,B,D)
#
#for p,color in zip([inter1,inter2],colors):
#	plt.plot(p[0],p[1],"o", color="black")
#
#
#
#
#plt.show()
#	