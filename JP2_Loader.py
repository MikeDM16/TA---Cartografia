#!/usr/bin/env python3

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob, cv2, time
import scipy.ndimage
from sys import getsizeof

class JP2_Loader():
	def __init__(self, path=''):
		self.directory = "../R10m/*.jp2"

	'''Function used to red the sentinel images from diferent sensores'''
	def read_sentinel_images(self):
		# Macro to avoid python to see pj2 size as DOS atacks
		Image.MAX_IMAGE_PIXELS = 1000000000
		
		directory = "../R10m/*.jp2"
		files = glob.glob(directory)

		start_time = time.time()

		for file_name in files: 
			if "TCI" in file_name:
				print("Reading TCI...")
				img_tci = scipy.ndimage.imread(file_name)
				#plt.imshow(img_red)
				#plt.show()

			'''
			if "B08" in file_name:
				print('Reading B08...')
				img_b08 = scipy.ndimage.imread(file_name)
				#plt.imshow(img_red)
				#plt.show()
			if "B04" in file_name:
				print('Reading B04...')
				img_red = scipy.ndimage.imread(file_name)
				#plt.imshow(img_red)
				#plt.show()
 
			if "B03" in file_name:
				print('Reading B03...')
				img_green = scipy.ndimage.imread(file_name)
				#plt.imshow(img_green)
				#plt.show()
			
			if "B02" in file_name:
				print('Reading B02...')
				img_blue = scipy.ndimage.imread(file_name)
				#plt.imshow(img_green)
				#plt.show()
			'''

		'''
		The TCI is an RGB image built from the B02 (Blue), B03 (Green), and B04 (Red) Bands. 
		Therefore there's no need of reading the 3 individual chanels 
		
		img_red = img_tci[:,:,0]
		img_green = img_tci[:,:,1]
		img_blue = img_tci[:,:,2]
		
		return (img_red, img_green, img_blue)
		'''
		
		print("Time taken to load sentinel bands: " + str(int(time.time() - start_time)) + " seg" )
		return (img_tci)

	'''Function used to build a RGB image from the B02 (Blue), B03 (Green), and B04 (Red) Bands '''
	def normalize_img(selg, img):
		# Normalize image values 
		max_pixel_value = img.max()
		img = np.multiply(img, 255.0)
		img = np.divide(img, max_pixel_value)
		img = img.astype(np.uint8)
		
		return img; 
	
	def mounting_all_together(self, img_red, img_green, img_blue):
		#print("Mounting all together...")
		
		# Create a 3 channel imagem by combine the individual RGB channels
		#img = np.dstack((img_red, img_green, img_blue)) 
		# OR
			
		(l,c) = img_red.shape
		img = np.zeros((l,c,3), 'uint16')
		img[:,:,0] = img_red
		img[:,:,1] = img_green
		img[:,:,2] = img_blue

		# Normalize image values 
		img = self.normalize_img(img)
		
		#mpimg.imsave('img_RGB.jpg', img, format='jpg')
		
		return img
	
	'''Function used to sabe an image in the jp2 format (16bit)'''
	def save_as_jps(self, image, dir = ""):
		image =  np.uint16(image);
		cv2.imwrite(dir + "teste.jp2", image)
	
