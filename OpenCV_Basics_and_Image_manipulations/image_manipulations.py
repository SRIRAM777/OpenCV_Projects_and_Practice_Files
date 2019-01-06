import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2

# Image Translation (Shifting the image from x and y axis)

image = cv2.imread("../../../../Downloads/msd7.jpeg")
height,width = image.shape[:2]
quarter_height,quarter_width = height/4,width/4

# Translation matrix [[1,0,Tx],[0,1,Ty]]

T_mat = np.float32([[1,0,quarter_width],[0,1,quarter_height]])
img_translation = cv2.warpAffine(image,T_mat,(width,height))
cv2.imshow('Translated image',img_translation)
cv2.waitKey(0)
cv2.destroyAllWindows()

## Image Rotation

## Rotation matrix [[cos_theta,-sin_theta],[sin_theta,cos_theta]]

image = cv2.imread("../../../../Downloads/msd7.jpeg")
height,width = image.shape[:2]
rot_mat = cv2.getRotationMatrix2D((width/2,height/2),77,1) # Center pts.,Angle of rotation,scale
rot_image = cv2.warpAffine(image,rot_mat,(width,height))
cv2.imshow('Rotated image',rot_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

## For removing the black space around the image
image = cv2.imread("../../../../Downloads/msd7.jpeg")
rot_image = cv2.transpose(image)
cv2.imshow('Rotated image',rot_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Image interpolation (A method of constructing new data pts within the range of known set of pts)
# cv2.INTER_AREA,cv2.INTER_NEAREST,cv2.INTER_LINEAR,cv2.INTER_CUBIC,cv2.INTER_LANCZOS4

image = cv2.imread("../../../../Downloads/msd7.jpeg")

img1 = cv2.resize(image,None,fx=0.5,fy=0.5)
cv2.imshow('Linear interpolation',img1)
cv2.waitKey(0)

img2 = cv2.resize(image,None,fx=0.5,fy=0.5,interpolation=cv2.INTER_CUBIC)
cv2.imshow('Cubic interpolation',img2)
cv2.waitKey(0)

img3 = cv2.resize(image,(400,400),interpolation=cv2.INTER_AREA)
cv2.imshow('Skewed interpolation',img3)
cv2.waitKey(0)

cv2.destroyAllWindows()

# Image Pyramids (Upscaling or Downscaling)
image = cv2.imread("../../../../Downloads/msd7.jpeg")
cv2.imshow('Smaller',cv2.pyrDown(image))
cv2.imshow('Larger',cv2.pyrUp(image))
cv2.waitKey(0)
cv2.destroyAllWindows()

# Image Cropping
image = cv2.imread("../../../../Downloads/msd7.jpeg")
height,width = image.shape[:2]
r1,c1 = int(height*.25),int(width*.25)
r2,c2 = int(height*.75),int(width*.75)
crop_img = image[r1:r2,c1:c2]
cv2.imshow('Cropped Image',crop_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Brightening and Darkening Images

image = cv2.imread("../../../../Downloads/msd7.jpeg")
one_arr = np.ones(image.shape,np.uint8) * 77
bright_img = cv2.add(image,one_arr)
dark_img = cv2.subtract(image,one_arr)
cv2.imshow('Bright Image',bright_img)
cv2.imshow('Dark Image',dark_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Bitwise operations and Masking

sq = np.zeros((300,300),np.uint8)
cv2.rectangle(sq,(50,50),(250,250),255,-2)
cv2.imshow('Square',sq)
cv2.waitKey(0)
ellipse = np.zeros((300,300),np.uint8)
cv2.ellipse(ellipse,(150,150),(150,150),30,0,180,255,-1)
cv2.imshow('Ellipse',ellipse)
cv2.waitKey(0)
cv2.destroyAllWindows()

and_img = cv2.bitwise_and(sq,ellipse)
cv2.imshow('And Image',and_img)
cv2.waitKey(0)
or_img = cv2.bitwise_or(sq,ellipse)
cv2.imshow('Or Image',or_img)
cv2.waitKey(0)
xor_img = cv2.bitwise_xor(sq,ellipse)
cv2.imshow('Xor Image',xor_img)
cv2.waitKey(0)
not_img = cv2.bitwise_not(sq,ellipse)
cv2.imshow('Not Image',not_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Convolutions and Blurring

# Blurring is an operation in which the pixels are averaged within a region

image = cv2.imread("../../../../Downloads/msd7.jpeg")
cv2.imshow('Original Image',image)

kernel_3x3 = np.ones((3,3),np.float32) / 9
blur_img1 = cv2.filter2D(image,-1,kernel_3x3)
cv2.imshow('3x3 Kernel Blurring',blur_img1)

kernel_7x7 = np.ones((7,7),np.float32) / 49
blur_img2 = cv2.filter2D(image,-1,kernel_7x7)
cv2.imshow('7x7 Kernel Blurring',blur_img2)
print(image[100,150])
print(blur_img1[100,150])
print(blur_img2[100,150])
cv2.waitKey(0)
cv2.destroyAllWindows()

# Other Blurring methods that are available
image = cv2.imread("../../../../Downloads/msd7.jpeg")
cv2.imshow('Original Image',image)
blur_img3 = cv2.blur(image,(3,3))
cv2.imshow('Averaging Blurring',blur_img3)

## Gaussian Fitering => Blurs an image using a Gaussian filter.
blur_img4 = cv2.GaussianBlur(image,(7,7),0) # (7,7) Kernel size. Ref: https://docs.opencv.org/3.1.0/d4/d86/group__imgproc__filter.html#gaabe8c836e97159a9193fb0b11ac52cf1
cv2.imshow('Gaussian Blurring',blur_img4)

## Median Blurring => The function smoothes an image using the median filter with the 𝚔𝚜𝚒𝚣𝚎×𝚔𝚜𝚒𝚣𝚎 aperture.
blur_img5 = cv2.medianBlur(image,5) # 5 is the aperture size
cv2.imshow('Median Blurring',blur_img5)

## Bilateral Filtering => Bilateral Filter can reduce unwanted noise very well while keeping edges fairly sharp. However, it is very slow compared to most filters.
blur_img6 = cv2.bilateralFilter(image,9,75,75) # 9 is the diameter used for filtering. Ref: https://docs.opencv.org/3.1.0/d4/d86/group__imgproc__filter.html#ga9d7064d478c95d60003cf839430737ed
cv2.imshow('Bilateral Blurring',blur_img6)
print(image[100,150])
print(blur_img3[100,150])
print(blur_img4[100,150])
print(blur_img5[100,150])
print(blur_img6[100,150])
cv2.waitKey(0)
cv2.destroyAllWindows()

# Image Denoising - Non Local Means Denoising

image = cv2.imread("../../../../Downloads/msd7.jpeg")
cv2.imshow('Original Image',image)
denoise_img = cv2.fastNlMeansDenoisingColored(image,None,6,6,7,21) # None is o/p image,7,21 is templateWindowSize and searchWindowSize, 6 is Parameter regulating filter strength for luminance component,6 is same as previous but for color components
cv2.imshow('Denoised Image',denoise_img)
print(image[100,150])
print(denoise_img[100,150])
cv2.waitKey(0)
cv2.destroyAllWindows()

# Sharpening => Strengthens edges in an image

# Kernel for sharpening => |-1,-1,-1|
#							|-1,9,-1|
#							|-1,-1,-1|

image = cv2.imread("../../../../Downloads/msd7.jpeg")
cv2.imshow('Original Image',image)
sharp_kernel = np.array([[-1,-1,-1],[-1,9,-1],[-1,-1,-1]])
sharp_img = cv2.filter2D(image,-1,sharp_kernel)
print(image[100,150])
print(sharp_img[100,150])
cv2.imshow('Sharpened Image',sharp_img)
cv2.waitKey(0)
cv2.destroyAllWindows()


# Thresholding (Binarization) - Making certain areas of image black or white

image = cv2.imread("../../../../Downloads/msd7.jpeg")
cv2.imshow('Original Image',image)
## Below 127 goes to 0(black) and above 127 goes to white(255)
ret,thresh1 = cv2.threshold(image,127,255,cv2.THRESH_BINARY)
cv2.imshow('Binary Threshold',thresh1)
cv2.waitKey(0)
cv2.destroyAllWindows()

## Opposite of the above method
ret,thresh2 = cv2.threshold(image,127,255,cv2.THRESH_BINARY_INV)
cv2.imshow('Binary Inverse Threshold',thresh2)
cv2.waitKey(0)
cv2.destroyAllWindows()

## Values above 127 are truncated are held at 127
ret,thresh3 = cv2.threshold(image,127,255,cv2.THRESH_TRUNC)
cv2.imshow('Truncated Threshold',thresh3)
cv2.waitKey(0)
cv2.destroyAllWindows()

## Values below 127 are set as 0
ret,thresh4 = cv2.threshold(image,127,255,cv2.THRESH_TOZERO)
cv2.imshow('To Zero Threshold',thresh4)
cv2.waitKey(0)
cv2.destroyAllWindows()

## Above 127 is set as 0
ret,thresh5 = cv2.threshold(image,127,255,cv2.THRESH_TOZERO_INV)
cv2.imshow('To Zero Inverse Threshold',thresh5)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Adaptive Thresholding Method => No need of mentioning the threshold value..Learns adaptively

image = cv2.imread("../../../../Downloads/msd7.jpeg")
cv2.imshow('Original Image',image)
ret,thresh1 = cv2.threshold(image,127,255,cv2.THRESH_BINARY)
cv2.imshow('Binary Threshold',thresh1)
cv2.waitKey(0)

## Gaussian blur removes any noise in the image if present
gray_img1 = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
image1 = cv2.GaussianBlur(gray_img1,(3,3),0)
thresh2 = cv2.adaptiveThreshold(image1,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,3,5)
cv2.imshow('Adaptive Mean Threshold',thresh2)
cv2.waitKey(0)

# thresh3 = cv2.adaptiveThreshold(gray_img1,0,255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)
# cv2.imshow("Otsu's Thresholding",thresh3)
# cv2.waitKey(0)

# thresh4 = cv2.adaptiveThreshold(gray_img1,0,255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)
# cv2.imshow("Gaussian Otsu's Thresholding",thresh4)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# Dilation and Erosion
# Dilation => Add pixels to the boundaries of objects in an image

image = cv2.imread("../../../../Downloads/msd7.jpeg")
cv2.imshow('Original Image',image)
cv2.waitKey(0)

kernel = np.ones((5,5),np.uint8)
dilation = cv2.dilate(image,kernel,iterations = 1)
cv2.imshow('Dilated Image',dilation)
cv2.waitKey(0)

## Erosion => Remove pixels at the boundaries of objects in an image
erosion = cv2.erode(image,kernel,iterations = 1)
cv2.imshow('Eroded Image',erosion)
cv2.waitKey(0)
cv2.destroyAllWindows()

## Opening and closing => Good for removing noise
## Opening => Erosion -> Dilation , Closing => Dilation -> Erosion

opening = cv2.morphologyEx(image,cv2.MORPH_OPEN,kernel)
cv2.imshow('Opening Image',opening)
cv2.waitKey(0)

closing = cv2.morphologyEx(image,cv2.MORPH_CLOSE,kernel)
cv2.imshow('Closing Image',closing)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Edge Detection and Image Gradients

# Edge Detection Algorithms
# 1. Sobel - Finds vertical or horizontal edges
# 2. Laplacian - Finds all orientations
# 3. Canny - Well defined edges and accurate detection. Optimal due to low error rate.

# Canny Edge Detection Algorithm 
# 1. Apply Gaussian Blurring
# 2. Find intensity gradient of image
# 3. Remove pixels that are not edges
# 4. Apply thresholds

image = cv2.imread("../../../../Downloads/msd7.jpeg")
cv2.imshow('Original Image',image)
cv2.waitKey(0)

## Horizontal edges using Sobel Detector
sobel_h = cv2.Sobel(image,cv2.CV_64F,0,1,ksize=5)
cv2.imshow('Horizontal Edges',sobel_h)
cv2.waitKey(0)

sobel_v = cv2.Sobel(image,cv2.CV_64F,1,0,ksize=5)
cv2.imshow('Vertical Edges',sobel_v)
cv2.waitKey(0)

sobel_or = cv2.bitwise_or(sobel_h,sobel_v)
cv2.imshow('Sobel OR',sobel_or)
cv2.waitKey(0)

lap = cv2.Laplacian(image,cv2.CV_64F)
cv2.imshow('Laplacian',lap)
cv2.waitKey(0)

canny = cv2.Canny(image,20,170)
cv2.imshow('Canny',canny)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Perspective and Affine Transforms (Similar to Cam Scanner app where we crop a part of a document and transform it properly)

image = cv2.imread("../../../../Downloads/msd7.jpeg")
cv2.imshow('Original Image',image)
cv2.waitKey(0)

points_1 = np.float32([[49,11],[205,15],[205,167],[49,167]])
points_2 = np.float32([[20,7],[211,17],[207,172],[30,181]])
mat = cv2.getPerspectiveTransform(points_1,points_2)
final_img = cv2.warpPerspective(image,mat,(300,300))
cv2.imshow('Final Image',final_img)
cv2.waitKey(0)
cv2.destroyAllWindows()


