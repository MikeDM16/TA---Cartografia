#!/usr/bin/env python3

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy.misc import toimage

class Processamento_Img():
	#-----------------------------------------------------------------------------------------------
	def processar(self, raster, opt):
		canny_img = self.canny(raster)
		
		morfologias = self.morfologias(canny_img, opt)
		
		res = [] # self.remover_zonas_clorofila(morfologias, raster)

		return(canny_img, morfologias, res)
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
	def morfologias(self, img, opt=0):
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
		'''
		erosion = cv2.erode(img,kernel,iterations = 1)
		opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel) # erosion followed by dilation
		'''
		
		dilation = cv2.dilate(img, kernel,iterations = 1)
		
		# Dilation followed by Erosion. 
		# It is useful in closing small holes inside the foreground objects, or small black points on the object.
		closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

		# difference between dilation and erosion of an image. The result will look like the outline of the object.
		gradient = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel) 

		kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(9,9))
		dilated = cv2.dilate(img, kernel)
		_, cnts, _ = cv2.findContours(dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		
		aux = cv2.drawContours(img, cnts, -1, (0,255,0), 3)
		_, ax = plt.subplots(2,3, figsize = (5,5))
		ax[0,0].imshow((img)) 
		ax[0,0].set_title(" Image")
		ax[0,1].imshow((dilated)) 
		ax[0,1].set_title("dilated")
		ax[0,2].imshow((aux))
		ax[0,2].set_title("contornos")

		if opt == 1 : plt.show()

		'''
		_, ax = plt.subplots(2,3, figsize = (5,5))
		ax[0,0].imshow(toimage(img)) 
		ax[0,0].set_title(" Image")
		ax[0,1].imshow(toimage(closing)) 
		ax[0,1].set_title("closing")
		ax[0,2].imshow(toimage(gradient))
		ax[0,2].set_title("gradient")

		ax[1,0].imshow(toimage(img - gradient))
		ax[1,0].set_title("img - gradient")
		ax[1,1].imshow(toimage(im_floodfill))
		ax[1,1].set_title("Floodfilled Image")
		ax[1,2].imshow(toimage(im_floodfill))
		ax[1,2].set_title("Floodfilled Image")
		plt.show()'''
		
		#if opt==1 : return cv2.bitwise_not(closing)
		#if opt==2 : 
		#	return  cv2.bitwise_not(gradient)
			
		return im_floodfill
	#-----------------------------------------------------------------------------------------------

	#-----------------------------------------------------------------------------------------------
	def remover_zonas_clorofila(self, morfologias, raster):
		l, c, _ = raster.shape

		res = np.ones((l,c)); #morfologias.copy() 

		for i in range(1,l-1):
			for j in range(1,c-1):
				# janela 3x3 para obter o valor RGB de um pixel 
				B = raster[i-1:i+1, j-1:j+1, 0].mean()
				G = raster[i-1:i+1, j-1:j+1, 1].mean()
				R = raster[i-1:i+1, j-1:j+1, 2].mean()

				# converter RGB para HSL 
				(H,S,L) = self.rgb2hsl(R,G,B)

				# Zonas verdes = clorofila != estradas (em teoria)
				if not ((H>50 and H<100) and (L > 70) and (S>100)):
					res[i,j] = 0 # limpar zonas a verde
	
		return res

	def rgb2hsl(self, R,G,B):
		var_R = ( R / 255 )
		var_G = ( G / 255 )
		var_B = ( B / 255 )

		H = S = L = 0

		var_Min = min( var_R, var_G, var_B )    # Min. value of RGB
		var_Max = max( var_R, var_G, var_B )    # Max. value of RGB
		del_Max = var_Max - var_Min             # Delta RGB value

		L = ( var_Max + var_Min )/ 2            # Lightness calculation

		if( del_Max == 0 ):                    # This is a gray, no chroma...
			H = 0
			S = 0   

		else:                                   # Chromatic data...
			if ( L < 0.5 ):
				S = del_Max / ( var_Max + var_Min )
			else:
				S = del_Max / ( 2 - var_Max - var_Min )

			del_R = ( ( ( var_Max - var_R ) / 6 ) + ( del_Max / 2 ) ) / del_Max
			del_G = ( ( ( var_Max - var_G ) / 6 ) + ( del_Max / 2 ) ) / del_Max
			del_B = ( ( ( var_Max - var_B ) / 6 ) + ( del_Max / 2 ) ) / del_Max
			
			if( var_R == var_Max ):
				H = del_B - del_G
			elif ( var_G == var_Max ):
				H = ( 1 / 3 ) + del_R - del_B
			elif ( var_B == var_Max ):
				H = ( 2 / 3 ) + del_G - del_R

		if ( H < 0 ): H += 1
		if ( H > 1 ): H -= 1

		H *= 240
		S *= 240
		L *= 240

		return (H,S,L)