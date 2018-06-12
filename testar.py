
import matplotlib.pyplot as plt

import cv2
import numpy as np

## Read
img = cv2.imread("sunflower.jpg")
img = cv2.imread("T29TNF_20180326T112109_TCI_10m.jp2")

## convert to hsv
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

## mask of green (36,0,0) ~ (70, 255,255)
mask = cv2.inRange(hsv, (36, 0, 0), (70, 255,255))

## slice the green
imask = mask>0
green = np.zeros_like(img, np.uint8)
green[imask] = img[imask]

## save 
cv2.imwrite("green.png", green)


'''
R_aux = np.zeros((kernel_size, kernel_size))
G_aux = np.zeros((kernel_size, kernel_size))
B_aux = np.zeros((kernel_size, kernel_size))
		
if(mode == 0):
R_aux = R[l:l+kernel_size, c:c+kernel_size]
	
elif(mode == 1):
G_aux = G[l:l+kernel_size, c:c+kernel_size]
	
elif(mode == 2):
	B_aux = B[l:l+kernel_size, c:c+kernel_size]
			
else:
	R_aux = R[l:l+kernel_size, c:c+kernel_size]
	G_aux = R[l:l+kernel_size, c:c+kernel_size]
	B_aux = R[l:l+kernel_size, c:c+kernel_size]
'''	