#!/usr/bin/env python3

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy.misc import toimage
import colorsys

class Processamento_Img():
	#-----------------------------------------------------------------------------------------------
	def processar(self, raster, opt):
		canny_img = self.canny(raster)
		
		res = self.remover_zonas_clorofila(raster, canny_img)
		
		morfologias, grad = self.morfologias(res, opt)
	
		return(morfologias, res, (255 - grad))
	#-----------------------------------------------------------------------------------------------
	
	#-----------------------------------------------------------------------------------------------
	def auto_canny(self, image, sigma=0.33):
		# compute the median of the single channel pixel intensities
		v = np.median(image)

		# apply automatic Canny edge detection using the computed median
		lower = int(max(0, (1.0 - sigma) * v))
		upper = int(min(255, (1.0 + sigma) * v))
		edged = cv2.Canny(image, lower, upper)

		# return the edged image
		return edged

	def canny(self, image):

		# Converting to gray scale...
		if len(image.shape) >= 3:
			#image = np.dot(image[...,:3], [0.2989, 0.5870, 0.1140])
			image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
		
		# Apply gaussian blur so that soft edges desapear 
		blur = cv2.GaussianBlur(image, (5, 5), 0)
		
		'''
		Apply Canny edge detection using a:
			- wide threshold, 
			- tight threshold, 
			- automatically determined threshold
		'''
		# wide = cv2.Canny(image, 10, 200)
		# tight = cv2.Canny(image, 225, 250)
		auto = self.auto_canny(blur)
		
		
		#wide = cv2.resize(wide, dsize=(500, 400), interpolation=cv2.INTER_CUBIC)
		#tight = cv2.resize(tight, dsize=(500, 400), interpolation=cv2.INTER_CUBIC)
		#auto = cv2.resize(auto, dsize=(800, 700), interpolation=cv2.INTER_CUBIC)
			
		# show the images
		#cv2.imshow("teste", np.hstack([wide, tight, auto]))
		#cv2.imshow("blur + Canny auto", np.hstack([auto]))
		#cv2.waitKey(0)
		#mpimg.imsave("img_Blur_canny_.jpeg", auto, format='jpg')
		
		#return(wide, tight, auto)
		return(auto)
	#-----------------------------------------------------------------------------------------------

	#-----------------------------------------------------------------------------------------------
	def morfologias(self, img, opt=1):
		# adaptiveThreshold( source_array, maxValue, Adaptive_Method, Threshold_Type, BlockSize, Constante substracted to the mean)
		im_th = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
		
		im_floodfill = im_th.copy()
	
		# Mask used to flood filling.
		# Notice the size needs to be 2 pixels than the image (3x3 kernel).
		h, w = im_th.shape[:2]
		mask = np.zeros((h+2, w+2), np.uint8)
		
		# Floodfill from point (0, 0)
		cv2.floodFill(im_floodfill, mask, (0,0), 255);
		
		# Invert floodfilled image
		im_floodfill_inv = cv2.bitwise_not(im_floodfill)
		
		# Combine the two images to get the foreground.
		im_out = im_th | im_floodfill_inv

		'''     
		# Display images.
		_, ax = plt.subplots(2,3, figsize = (5,5))
		ax[0,0].imshow(toimage(img)) 
		ax[0,0].set_title(" Image")
		ax[0,1].imshow(toimage(im_th)) 
		ax[0,1].set_title("Thresholded Image")
		ax[0,2].imshow(toimage(im_floodfill))
		ax[0,2].set_title("Floodfilled Image")
		
		ax[1,0].imshow(toimage(im_floodfill_inv))
		ax[1,0].set_title("Inverted Floodfilled Image")
		ax[1,1].imshow(toimage(im_out))
		ax[1,1].set_title("Foreground")
		ax[1,2].imshow(toimage(im_out))
		ax[1,2].set_title("Foreground")
		plt.show()'''
		

		kernel = np.ones((5,5),np.uint8)
		
		#erosion = cv2.erode(img,kernel,iterations = 1)
		#opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel) # erosion followed by dilation
		#dilation = cv2.dilate(img, kernel,iterations = 1)
		
		# Dilation followed by Erosion. 
		# It is useful in closing small holes inside the foreground objects, or small black points on the object.
		closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

		# difference between dilation and erosion of an image. The result will look like the outline of the object.
		gradient = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel) 

		'''
		# Find contours on the image (similar to gradient)
		kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(9,9))
		dilated = cv2.dilate(img, kernel)
		_, cnts, _ = cv2.findContours(dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		'''
		aux = cv2.morphologyEx(im_floodfill, cv2.MORPH_GRADIENT, kernel)

		if opt == 1 : 
			_, ax = plt.subplots(2,3, figsize = (5,5))
			ax[0,0].imshow(toimage(img)) 
			ax[0,0].set_title(" Image")
			ax[0,1].imshow(toimage(closing)) 
			ax[0,1].set_title("closing")
			ax[0,2].imshow(toimage(gradient))
			ax[0,2].set_title("gradient")

			ax[1,0].imshow((aux))
			ax[1,0].set_title("grad im_flood ")
			ax[1,1].imshow(toimage(im_floodfill))
			ax[1,1].set_title("Floodfilled Image")
			ax[1,2].imshow(toimage(im_floodfill_inv))
			ax[1,2].set_title("Floodfilled inv")
			plt.show()
					
		return im_floodfill, gradient
	#-----------------------------------------------------------------------------------------------

	#-----------------------------------------------------------------------------------------------
	def remover_zonas_clorofila(self, raster, canny):
		l, c, _ = raster.shape
		res = canny.copy() 
		#res = np.zeros((l,c)); #morfologias.copy() 
		
		## convert to hsv
		hsv = cv2.cvtColor(raster, cv2.COLOR_BGR2HSV)

		## mask of green (36,0,0) ~ (70, 255,255)
		mask = cv2.inRange(hsv, (36, 50, 20), (70, 255,255))

		id_l, id_c = np.nonzero( mask > 0)
		for i,j in zip(id_l, id_c):
			res[i,j] = 1


		'''
		## slice the green
		green_mask = (mask > 0)
		
		green = np.zeros_like(raster, np.uint8)
		green[green_mask] = raster[green_mask]
		
		## save
		cv2.imwrite("green.png", green)''' 
		return res
