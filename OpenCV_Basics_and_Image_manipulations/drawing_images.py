import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2

# image1 = np.zeros((512,512,3),np.uint8)
# image2 = np.zeros((512,512),np.uint8)

# cv2.imshow('Black (Color)',image1)
# cv2.imshow('Black (B&W)',image2)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

## Line

image1 = np.zeros((512,512,3),np.uint8)
cv2.line(image1,(0,0),(511,511),(255,127,0),5)
cv2.line(image1,(0,511),(511,0),(255,127,0),5)
cv2.imshow('Line on the image',image1)
cv2.waitKey(0)
cv2.destroyAllWindows()

## Rectangle

image1 = np.zeros((512,512,3),np.uint8)
cv2.rectangle(image1,(50,50),(150,150),(255,127,0),5)
cv2.imshow('Line on the image',image1)
cv2.waitKey(0)
cv2.destroyAllWindows()

## Circle

image1 = np.zeros((512,512,3),np.uint8)
cv2.circle(image1,(300,300),100,(255,127,0),5)
cv2.imshow('Line on the image',image1)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Polygon
image1 = np.zeros((512,512,3),np.uint8)
pts = np.array([[10,50],[400,50],[90,200],[50,500]],np.int32)
pts = pts.reshape((-1,1,2))
cv2.polylines(image1,[pts],True,(0,0,255),3)
cv2.imshow('Polygon Image',image1)
cv2.waitKey(0)
cv2.destroyAllWindows()

## Adding text to image
image1 = np.zeros((512,512,3),np.uint8)
cv2.putText(image1,'Sriram',(75,290),cv2.FONT_HERSHEY_DUPLEX,3,(255,127,0),5)
cv2.imshow('Polygon Image',image1)
cv2.waitKey(0)
cv2.destroyAllWindows()




