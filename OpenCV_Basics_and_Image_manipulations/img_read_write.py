import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt

## Reading an Image

image = cv2.imread("../../../../Downloads/msd7.jpeg")
print(image.shape)

image1 = cv2.imread("../../../../Downloads/msd7.jpeg")
gray_img1 = cv2.cvtColor(image1,cv2.COLOR_BGR2GRAY)

## Saving an Image
cv2.imwrite('grayimg1.jpg',gray_img1)
cv2.imshow('Grey image',gray_img1)
cv2.waitKey(0)
cv2.destroyAllWindows()

## Alternate way to convert image to grayscale directly


image1 = cv2.imread("../../../../Downloads/msd7.jpeg",0) ==> Note the zero in the end. 
cv2.imshow('Grey image',gray_img1)
cv2.waitKey(0)
cv2.destroyAllWindows()

## BGR to HSV (Hue,Saturation,Value)

image1 = cv2.imread("../../../../Downloads/msd7.jpeg")
gray_img1 = cv2.cvtColor(image1,cv2.COLOR_BGR2HSV)
cv2.imshow('Grey image1',gray_img1)
cv2.imshow('Grey image2',gray_img1[:,:,0])
cv2.imshow('Grey image3',gray_img1[:,:,1])
cv2.imshow('Grey image4',gray_img1[:,:,2])
cv2.waitKey(0)
cv2.destroyAllWindows()

## Exploring individual channels in a RGB image

image1 = cv2.imread("../../../../Downloads/msd7.jpeg")
b,g,r = cv2.split(image1)
cv2.imshow('B',b)
cv2.imshow('G',g)
cv2.imshow('R',r)
cv2.imshow('Original Image',cv2.merge([b,g,r]))
cv2.imshow('Amplified Image',cv2.merge([b+100,g+50,r+70]))
cv2.waitKey(0)
cv2.destroyAllWindows()

image1 = cv2.imread("../../../../Downloads/msd7.jpeg")
b,g,r = cv2.split(image1)
zeros = np.zeros(image1.shape[:2],dtype='uint8')
cv2.imshow('Red Image',cv2.merge([zeros,zeros,r]))
cv2.imshow('Green Image',cv2.merge([zeros,g,zeros]))
cv2.imshow('Blue Image',cv2.merge([b,zeros,zeros]))
cv2.waitKey(0)
cv2.destroyAllWindows()





