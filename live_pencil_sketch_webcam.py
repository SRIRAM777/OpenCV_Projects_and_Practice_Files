import cv2
import numpy as np
import matplotlib.pyplot as plt

def sketch(image):
	## Convert image to Grayscale
	gray_img1 = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

	## Remove noise from the image
	blur_img = cv2.GaussianBlur(gray_img1,(5,5),0)

	## Detect edges in the image using Canny Algorithm
	can_img = cv2.Canny(blur_img,10,100)

	##Perform Binary Inverse Threshold on the image
	ret,thresh1 = cv2.threshold(can_img,70,250,cv2.THRESH_BINARY_INV)

	return thresh1


cap = cv2.VideoCapture(0)
while(True):
    ret, frame = cap.read()
    cv2.imshow('Live Pencil Sketcher',sketch(frame))
    if cv2.waitKey(1) & 0xFF == ord('q'):
    	break

cap.release()
cv2.destroyAllWindows() 
cv2.waitKey(0)

