from scipy.misc import imread, imshow
from scipy.ndimage import label
from scipy import asarray, ones, vstack, hstack
from sys import stdout
#i m working parallaly
import cv2
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
from PIL.ImageFilter import (
    MinFilter
    )

img = cv2.imread('image.jpg')


#median filtering
median = cv2.medianBlur(img,5)



plt.subplot(121),plt.imshow(img),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(median),plt.title('Blurred')
plt.xticks([]), plt.yticks([])
plt.show()


#step2
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imwrite('img.jpg',gray)
 
cv2.imshow('Original image',img)
cv2.imshow('Gray image', gray)
 
cv2.waitKey(0)
cv2.destroyAllWindows()

#step3
img = cv2.imread('img.jpg')
img_to_yuv = cv2.cvtColor(img,cv2.COLOR_BGR2YUV)
img_to_yuv[:,:,0] = cv2.equalizeHist(img_to_yuv[:,:,0])
hist_equalization_result = cv2.cvtColor(img_to_yuv, cv2.COLOR_YUV2BGR)
cv2.imshow('result.jpg',hist_equalization_result)
 
cv2.imwrite('result.jpg',hist_equalization_result)

#step4

img1 = cv2.imread('img.jpg',0) 
# create a CLAHE object (Arguments are optional).
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
cl1 = clahe.apply(img1) 
cv2.imwrite('step.jpg',cl1)

#step5
# Read Image1 
imageA = cv2.imread('result.jpg') 
  
# Read image2 
imageB = cv2.imread('step.jpg') 
  
# Add the images 
blend = cv2.add(imageA, imageB) 
  
# Show the image 
cv2.imshow('blendimage', blend)
cv2.imwrite('blendimage.jpg',blend)

# Wait for a key 
cv2.waitKey(0) 
  
# Distroy all the window open 
cv2.destroyAllWindows()

#step6
img1 = cv2.imread('blendimage.jpg') 
  
# Read image2 
img2 = cv2.imread('result.jpg') 

# sub the images 
sub = cv2.subtract(img1, img2)
cv2.imwrite('sub.jpg',sub) 


#step7
image1 = cv2.imread('blendimage.jpg') 
  
# Read image2 
image2 = cv2.imread('sub.jpg') 
  
# Add the images 
addimg = cv2.add(image1, image2)
cv2.imwrite('addimg.jpg',addimg)

#step8
simg = Image.open('addimg.jpg')
dimg = simg.filter(MinFilter(size=9))
dimg.save("filter.jpg")

#step910
imgt = cv2.imread('filter.jpg',0)
 
ret, imgf = cv2.threshold(imgt, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
cv2.imwrite('imgt.jpg',imgf)
 
plt.subplot(3,1,1), plt.imshow(imgt,cmap = 'gray')
plt.title('Original Noisy Image'), plt.xticks([]), plt.yticks([])
plt.subplot(3,1,2), plt.hist(img.ravel(), 256)
plt.axvline(x=ret, color='r', linestyle='dashed', linewidth=2)
plt.title('Histogram'), plt.xticks([]), plt.yticks([])
plt.subplot(3,1,3), plt.imshow(imgf,cmap = 'gray')
plt.title('Otsu thresholding'), plt.xticks([]), plt.yticks([])
plt.show()

#step11

imgm = cv2.imread('imgt.jpg', 0) 
kernel = np.ones((3,3), np.uint8) 
img_erosion = cv2.erode(imgm, kernel, iterations=1) 
img_dilation = cv2.dilate(imgm, kernel, iterations=1) 
  
cv2.imshow('Input', imgm) 
cv2.imwrite('Erosion.jpg', img_erosion) 
cv2.imwrite('Dilation.jpg', img_dilation) 
  
cv2.waitKey(0) 

#step12
bw= cv2.imread('Dialation.jpg', 0)
ret, labels = cv2.connectedComponents(bw)


#labeled, nr_objects = ndimage.label(bw)
#print 'Nr objects:', nr objects
#labeled, n = label(bw, ones((3,3),np.uint8))
#objects_per_unit_area = n * 1.0 / labeled.size
#eight, width = bw.shape
#pad_south = 255 * ones((1, width))
#pad_east = 255 * ones((height + 1, 1))
#bw_padded = hstack((vstack((bw, pad_south)), pad_east))

#stdout.write("objects per unit area (naive):    %.4f\n" % objects_per_unit_area)
#stdout.write("objects per unit area (unbiased): %.4f\n" % objects_per_unit_area2)

 







