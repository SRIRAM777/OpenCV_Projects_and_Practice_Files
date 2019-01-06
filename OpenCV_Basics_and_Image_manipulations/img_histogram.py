import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2

image = cv2.imread("../../../../Downloads/msd7.jpeg")

## Histogram of an image (Frequencies of each pixel values)

#cv2.calcHist(images, channels, mask, histSize, ranges[, hist[, accumulate]])
# 1. images => path of image , 2. channels => [0] for gray scale, [0],[1],[2] for BGR, 
# 3. mask => finding hist for whole image or part of the img. 4.histsize => Bin count [256], 5. range => Range of values

histogram = cv2.calcHist([image],[0],None,[256],[0,256])
print(histogram[0:5])

plt.hist(image.ravel(),256,[0,256])
plt.show()

color = ('b','g','r')

for i,col in enumerate(color):
	histogram1 = cv2.calcHist([image],[i],None,[256],[0,256])
	plt.hist(histogram1,color=col)
	plt.xlim([0,256])
plt.show()
