import cv2
import numpy as np 

sampleNum = 0

uid = input('enter user id')

cam = cv2.VideoCapture(0)

while(True):
	ret,img = cam.read()	#ret ois used to find if the camera is providing the frames or not.....we can ignore this with "_"
	gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	x=20
	y=100
	w=300
	h=250
	sampleNum+=1
	#creates the gtreen rectangle
	cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
	img2=img[y:y+h,x:x+w]	

	#edge detection(canny edge detection) 	
	edges = cv2.Canny(img2 , 100, 200)

	#gradiants
	laplacian = cv2.Laplacian(img2, cv2.CV_64F)


	#thresholding
	ret, threshold = cv2.threshold(img2,12,255,cv2.THRESH_BINARY)
	grayscaled = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
	ret, threshold2 = cv2.threshold(grayscaled, 100,255,cv2.THRESH_BINARY)
	gaus = cv2.adaptiveThreshold(grayscaled, 255 ,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,115,1)

	#cv2.imshow("threshold",threshold)
	#cv2.imshow("threshold2",threshold2)
	cv2.imshow("adaptive threshold", gaus)


	cv2.imwrite('data set1/'+str(uid)+'_'+str(sampleNum)+'.jpg',gray[y:y+h,x:x+w])
	cv2.imwrite('data set2/'+str(uid)+'_'+str(sampleNum)+'.jpg',gaus[y:y+h,x:x+w])



	cv2.waitKey(100)	#there is a gap of 100 miliseconds between every frame caputured

	cv2.imshow('ROI',img)
	#cv2.imshow("Edges", edges)
	#cv2.imshow("laplacian", laplacian)
	cv2.waitKey(1)
	if(sampleNum>50):
		break
cam.release()
cam.destroyAllWindows()
