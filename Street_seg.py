#!/usr/bin/env python3

import JP2_Loader
import Processamento_Img
import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np

def main():
	print("Sentinel 2 Spacial Resolution Bands:\n" + 
		"   Band 2 - Blue (490nm)\t Band 3 - Green (560nm)\t Band 4 - Orange (665nm)\n" + 
		"   TCI - RGB image built from the B02 (Blue), B03 (Green) and B04 (Red) Bands\n")

	jp2_interface = JP2_Loader.JP2_Loader()
	process_img = Processamento_Img.Processamento_Img()
	
	(raster) = jp2_interface.read_sentinel_images()

	(max_h, max_w, _) = raster.shape
	size_w = 400 # size of the windows displayed 
	
	shift_size = 50 # step to navigate through image
	kernel_size = 400 # kernel size to exctract sub region of image

	# Initial position 
	l = 0; 	c = 3000

	# initial options 
	mode = 3; opt = 0;

	while(True):
		raster_aux = np.zeros((kernel_size,kernel_size,3)).astype(np.uint8)
		#b08_aux = b08[l:l+kernel_size, c:c+kernel_size].astype(np.uint8)
		
		if(mode == 0):
			raster_aux[:,:,0] = raster[l:l+kernel_size, c:c+kernel_size][:,:,0]
			print("red")
		elif(mode == 1):
			raster_aux[:,:,1] = raster[l:l+kernel_size, c:c+kernel_size][:,:,1]
			print("green")
		elif(mode == 2):
			raster_aux[:,:,2] = raster[l:l+kernel_size, c:c+kernel_size][:,:,2]
			print("azul")
		else:
			raster_aux = raster[l:l+kernel_size, c:c+kernel_size]
		
		raster_aux = cv2.resize(raster_aux, dsize=(600, 600), interpolation=cv2.INTER_CUBIC)
		#b08_aux = cv2.resize(b08_aux, dsize=(size_w, size_w), interpolation=cv2.INTER_CUBIC)
		
		(morfologias, res, grad) = process_img.processar(raster_aux, opt)

		# Resize results to display 
		raster_aux = cv2.resize(raster_aux, dsize=(size_w, size_w), interpolation=cv2.INTER_CUBIC)
		morfologias = cv2.resize(morfologias, dsize=(size_w, size_w), interpolation=cv2.INTER_CUBIC)
		res = cv2.resize(res, dsize=(size_w, size_w), interpolation=cv2.INTER_CUBIC)
		grad = cv2.resize(grad, dsize=(size_w, size_w), interpolation=cv2.INTER_CUBIC)

		# Display results 
		cv2.namedWindow('Regiao')        # Create a named window
		cv2.moveWindow('Regiao', 100,300)  # Move it to somewhere 
		cv2.imshow("Regiao",  raster_aux)
					
		cv2.namedWindow("segmentation")        # Create a named window
		cv2.moveWindow("segmentation", 100 + (size_w + 10),300)  # Move it to somewhere  
		cv2.imshow("segmentation",  morfologias)
		
		cv2.namedWindow("Grad")   # Create a named window
		cv2.moveWindow("Grad", 100 + 2*(size_w + 10),300)  # Move it to somewhere  
		cv2.imshow("Grad",  grad)
		
		# Wait for input option... 
		k = 0xFF & cv2.waitKey(0)

		# moving along the image 
		if (k == ord('d')): # right key ->
			c = min( c + shift_size, max_w)

		elif (k == ord('a')): # left key <- 
			c = max( c - shift_size, 0)

		elif (k == ord('w')): # up key ^
			l = max( l - shift_size, 0)

		elif (k == ord('s')): # down key v 
			l = min( l + shift_size, max_h)

		# tweaking zoom 
		elif (k == ord('+')): # down key v 
			kernel_size = max( kernel_size - 50, 50)
		
		elif (k == ord('-')): # down key v 
			kernel_size = min( kernel_size + 50, max_h)

		
		elif(k == ord('1')): opt = 1 # use only red chanel
		elif(k == ord('2')): opt = 2 # use only green chanel
		elif(k == ord('3')): opt = 0 # use only Blue chanel
		elif(k == ord('4')): mode = (3) # use only RGB chanel
		

		if k == ord('q'):
			cv2.destroyAllWindows()
			break
			

	return 
	
	'''
	#(height, width, channels) = img.shape
	#height = int(height/5)
	#vwidth  = int(width/5)
	print("Operar sobre raster com " + str(height) + "x" + str(width))

	resultado = np.zeros((height, width))
	kernel_size = 200
	iter = 0
	print(list( range(0, height, kernel_size)))

	lx = 0
	_, ax = plt.subplots(3,2, figsize = (7,7))
	for l in range(0, height, kernel_size):
		cx = 0
		for c in range(0, width, kernel_size):	
			R_aux = R[l:l+kernel_size, c:c+kernel_size]
			G_aux = G[l:l+kernel_size, c:c+kernel_size]
			B_aux = B[l:l+kernel_size, c:c+kernel_size]
			(h,w) = R_aux.shape

			img = jp2_interface.mounting_all_together(R_aux, G_aux, B_aux)
			iter += 1
			canny_img = process_img.canny(img)
			morfologias = process_img.morfologias(canny_img)

			res = cv2.resize(morfologias, dsize=(h, w), interpolation=cv2.INTER_CUBIC)
			#cv2.imshow("Final resize", morfologias)
			#cv2.waitKey(0)

			resultado[l:l+kernel_size, c:c+kernel_size] = res
			
			ax[lx,cx].imshow((res)) 
			ax[lx,cx].set_title("Thresholded Image" + str(cx))
			cx = cx + 1 
		lx = lx +1
	
	ax[2,0].imshow((imgIni)) 
	ax[2,0].set_title(" Image")
	
	ax[2,1].imshow((resultado)) 
	ax[2,1].set_title(" Image")
	plt.show()

	jp2_interface.save_as_jps(resultado)
	'''

if __name__ == "__main__": main()