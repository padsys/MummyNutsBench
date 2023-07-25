
# Copyright (c) 2023, University of California, Merced. All rights reserved.

# This file is part of the MummyNutsBench software package developed by
# the team members of Prof. Xiaoyi Lu's group (PADSYS Lab) at the University
# of California, Merced.

# For detailed copyright and licensing information, please refer to the license
# file LICENSE in the top level directory.


# Takes yolo annotations and rotates/stretches image and annotation boxes
# Still WIP, the stretching math needs some work

import cv2, numpy as np, os, math, csv

# angles
angUpper = 181		# upper bounds of rotation angle
angLower = 0		# lower bounds of rotation angle
angIncrement = 180	# increment of rotation angle
					# same below v
# scale percent width
spwUpper = 151		# set 1 higher to make sure it is included
spwLower = 100
spwIncrement = 50

# scale percent height
sphUpper = 151
sphLower = 100
sphIncrement = 50

# paths
nutsPath = "C:\\Users\\colin\\Desktop\\Sky_BG\\"
outputPath = "C:\\Users\\colin\\Desktop\\transformed\\"
pics = os.listdir(nutsPath)

def makeDot(img, corners, color):
	for x in corners:
		img[x[1], x[0]] = color
		img[x[1]+1, x[0]] = color
		img[x[1], x[0]+1] = color
		img[x[1]-1, x[0]] = color
		img[x[1], x[0]-1] = color
	return img	

def rotate(XY, origin, angle):
	x = XY[0] * math.cos(math.radians(angle)) - XY[1] * math.sin(math.radians(angle)) + origin[0] - origin[0] * math.cos(math.radians(angle)) + origin[1] * math.sin(math.radians(angle))
	y = XY[0] * math.sin(math.radians(angle)) + XY[1] * math.cos(math.radians(angle)) + origin[1] - origin[0] * math.sin(math.radians(angle)) - origin[1] * math.cos(math.radians(angle))
	x = round(x)
	y = round(y)
	return [x,y]

def getNumFromAnnotation(annotate):
	index = 2
	prev = 2
	numOut = []

	while index < len(annotate):
		index = annotate.find(' ', index)
		if index == -1:
			numOut.append(float(annotate[prev:]))
			break
		numOut.append(float(annotate[prev:index]))
		prev = index
		index += 1

	return numOut


with open(outputPath + '_annotations.csv', 'w', newline='') as file:
	writer = csv.writer(file)
	writer.writerow(["filename","width","height","class","xmin","ymin","xmax","ymax"])

for pic in pics:
	if pic[-4:] == ".JPG":
		original = cv2.imread(nutsPath + "\\" + pic)
		annotate = open(nutsPath + pic[:-4] + ".txt", "r").read()
		for ang in range(angLower, angUpper, angIncrement):
			for spw in range(spwLower, spwUpper, spwIncrement):
				for sph in range(sphLower, sphUpper, sphIncrement):
					if spw != sph or spw == 100:
						try:
							img = original.copy()
							imgName = pic[:-4] + "_A-" + str(ang) + "_W-" + str(spw) + "_H-" + str(sph)

							# 	Corner array representations
							#	[0,] = TL
							# 	[1,] = TR
							# 	[2,] = BL
							#	[3,] = BR
							#	[,0] = x
							#	[,1] = y
							#			TL  ,  TR  ,  BL  ,  BR
							corners = [[0,0], [0,0], [0,0], [0,0]]

							# Set new width/height
							oldWidth = img.shape[1]
							oldHeight = img.shape[0]
							newWidth = int(oldWidth * spw / 100)
							newHeight = int(oldHeight * sph / 100)

							# Setup annotation array
							# [originX, originY, width, height]
							annotateNums = getNumFromAnnotation(annotate)
							origin = [(annotateNums[0] * newWidth), (annotateNums[1] * newHeight)]
							innerWidth = annotateNums[2] * newWidth
							innerHeight = annotateNums[3] * newHeight

							# Fill in corners array
							corners[0][0] = int(round(origin[0] - innerWidth/2))
							corners[1][0] = int(round(origin[0] + innerWidth/2))
							corners[2][0] = int(round(origin[0] - innerWidth/2))
							corners[3][0] = int(round(origin[0] + innerWidth/2))
							corners[0][1] = int(round(origin[1] + innerHeight/2))
							corners[1][1] = int(round(origin[1] + innerHeight/2))
							corners[2][1] = int(round(origin[1] - innerHeight/2))
							corners[3][1] = int(round(origin[1] - innerHeight/2))

							# Rotate corners
							for i, x in enumerate(corners):
								corners[i] = rotate(x, [newWidth/2, newHeight/2], ang)

							# Fix corners to be parallel with respictive axis after any rotations
							# May need add math to shave the distances for angles 
							#   that would make the box bigger than it needs to be Ex. 45 deg
							xMax = 0
							yMax = 0
							for corner in corners:
								if corner[0] > xMax:
									xMax = corner[0]
								if corner[1] > yMax:
									yMax = corner[1]

							xMin = int(2*origin[0] - xMax)
							yMin = int(2*origin[1] - yMax)


							# Apply corner fixes
							corners[0][0] = xMin
							corners[1][0] = xMax
							corners[2][0] = xMin
							corners[3][0] = xMax

							corners[0][1] = yMax
							corners[1][1] = yMax
							corners[2][1] = yMin
							corners[3][1] = yMin

							# Stretch then Rotate image
							img = cv2.resize(img, (newWidth, newHeight))
							mat = cv2.getRotationMatrix2D((newWidth/2,newHeight/2), ang, 1.0)
							img = cv2.warpAffine(img, mat, (newWidth, newHeight))

							# Write image
							#img = makeDot(img, corners, [0,0,255])
							cv2.imwrite(outputPath + imgName + ".jpg", img)

							# CSV Annoations (easier to do csv than yolo annotations)
							# Specifically ("Tensorflow Object Detection-CSV") annotation
							# ["filename","width","height","class","xmin","ymin","xmax","ymax"]
							newAnnotation = [imgName+".jpg", newWidth, newHeight, "nut", xMin, yMin, xMax, yMax]
							stringAnnotation = ""
							for x in newAnnotation:
								stringAnnotation += str(x)
								stringAnnotation += ","
							stringAnnotation = stringAnnotation[:-1] + "\n"

							with open(outputPath + '_annotations.csv','a') as file:
								file.write(stringAnnotation)



							# YOLO Annotations
							# Harder to do since more math is needed to convert back to yolo format
							'''newAnnotations = ["0 "]
							newAnnotations.append(str(origin[0]/newWidth) + " ")
							newAnnotations.append(str(origin[1]/newHeight) + " ")
							newAnnotations.append(str(2*(yMax-origin[1])/newHeight) + " ")
							newAnnotations.append(str(2*(xMax-origin[0])/newWidth))

							createAnnotation = open(outputPath + imgName + ".txt", "w+")
							createAnnotation.write(newAnnotations[0] + newAnnotations[1] + newAnnotations[2] + newAnnotations[3] + newAnnotations[4])
							createAnnotation.close()'''

						except:
							print("Error with: " + imgName)