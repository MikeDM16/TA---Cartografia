#!/usr/bin/env python3

from PIL import Image; 
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import glob, os, sys
import cv2
import skimage
from skimage import filters, feature, color
from scipy import fftpack, ndimage, signal
from scipy.misc import toimage
import math 
import mahotas

#-----------------------------------------------------------------------------------------------
#   Função de Labeling. 
#   - Recebe como argumento a imagem sobre a qual extrair os componentes ligados 
#   - Devolve o conjunto de elementos ligados de uma imagem, juntamente o número de 
# objetos encontrados  
def labeling(img):
    # Determinar o sub conjunto de objetos da imagem
    labeling, nr_objetos = mahotas.label(img)
    return labeling, nr_objetos
#-----------------------------------------------------------------------------------------------

#-----------------------------------------------------------------------------------------------
#   Função para remoção de regiões ligadas. 
#   - Recebe como argumento uma lista com o conjunto de componentes ligadas de uma imagens pré processada
#   - Neste contexto remove apenas o fundo da imagem. ''' 
def remove_regions(labeling):
    # Determinar o tamanho de cada objeto da imagem 
    tamanhos = mahotas.labeled.labeled_size(labeling)
    # Ordenar os objetos retornados por tamanho 
    tamanhos.sort()
    
    # Remover os componentes ligados com tamanho superior ao do fundo
    t_borda = tamanhos[1]
    fundo = np.where(tamanhos > t_borda)
    removidos = mahotas.labeled.remove_regions(labeling, fundo)
    return removidos 

#-----------------------------------------------------------------------------------------------

def filtro_Gaussian(img, D0 = 10):
	image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	# Obter os parametros de padding
	M, N = img.shape
	P = 2*M
	Q = 2*N
	
	#Imagem extendida (padded) da imagem inicial
	f_pad = np.zeros((P, Q), dtype=float)
	f_pad[0:M, 0:N] = img
		
	# Ciclo para centrar a imagem 
	# Para optimizar usar só o 1º quadrante MxN
	for i in range(M):
		for j in range(N):
			f_pad[i,j] = f_pad[i,j] * pow( -1, i+j) 
	
	# Calculo da transformada DFT 
	F = fftpack.fft2(f_pad) 
	
	# Geração filtro atraves da função H 
	H = np.zeros((P, Q), dtype=float)
	for u in range(P):
		for v in range(Q):
			d_uv = pow( ((u - P/2)**2 + (v-Q/2)**2), 1/2) 
			e = (d_uv**2)/((2*D0)**2 + 1e-5)
			H[u,v] = np.exp(-e)
			
	# Funcao G
	G = F * H
	
	# Imagem processada
	G_inv = fftpack.ifft2(G).real #Já passa de complexo paea real
	g_processada = np.zeros((P, Q), dtype=float)
	for u in range(P):
		for v in range(Q):
			g_processada[u,v] = G_inv[u,v] * pow((-1), (u+v))
	
	''' 
	_, ax = subplots(1,3,figsize = (12,12))
	ax[0].imshow(log10(abs(H)+1e-5))
	ax[0].set_title("Espectro do filtro butterworth band reject H")
	ax[1].imshow(log10(abs(F)+1e-5))
	ax[1].set_title("Espectro da transformada F (DFT)")
	ax[2].imshow(log10(abs(G)+1e-5))
	ax[2].set_title("Espectro do filtro real e simetrico G")
	''' 
	
	return g_processada[0:M, 0:N]
#-----------------------------------------------------------------------------------------------
#   Função de binarização. 
#   - Recebe como argumento a imagem a binarizar. 
#   - Retorna a matriz binária da imagem binarizada e o valor de corte 
# utilizado como threshold 
def binarizacao(img):
	# Converter argumento de entrada para imagem em nivel de cinzentos, caso não o seja
	# matplotlib.pyplot.gray()
	# Inicialização da matriz para a imagem binarizada a zeros
	bin = np.zeros_like(img)
	
	# Valor a aplicar como treshold à imagem
	threshold_otsu = skimage.filters.threshold_otsu(img.astype('uint8'))
	
	#   Nas posições da imagem inicial onde a intensidade do pixel seja 
	# superior à intensidade de corte definida pelo threshold, o pixel 
	# toma o valor de 1 (branco)
	for i in range(img.shape[0]):
		for j in range(img.shape[1]):
			if(img[i,j] > threshold_otsu):
				bin[i,j] = 1
	
	#bin = img > threshold_otsu
	return bin

#-----------------------------------------------------------------------------------------------
def auto_canny(image, sigma=0.33):
	# compute the median of the single channel pixel intensities
	v = np.median(image)

	# apply automatic Canny edge detection using the computed median
	lower = int(max(0, (1.0 - sigma) * v))
	upper = int(min(255, (1.0 + sigma) * v))
	edged = cv2.Canny(image, lower, upper)

	# return the edged image
	return edged

def canny(image, opt=0):
	# Converting to gray scale...
	#image = np.dot(image[...,:3], [0.2989, 0.5870, 0.1140])
	image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	# Apply gaussian blur so that soft edges desapear 
	blur = cv2.GaussianBlur(image, (5, 5), 0)
	# apply Canny edge detection using a:
	# 	- wide threshold, 
	# 	- tight threshold, 
	# 	- automatically determined threshold
	# wide = cv2.Canny(image, 10, 200)
	# tight = cv2.Canny(image, 225, 250)
	auto = auto_canny(blur)
	#wide = cv2.resize(wide, dsize=(500, 400), interpolation=cv2.INTER_CUBIC)
	#tight = cv2.resize(tight, dsize=(500, 400), interpolation=cv2.INTER_CUBIC)
	auto = cv2.resize(auto, dsize=(800, 700), interpolation=cv2.INTER_CUBIC)
	#image = cv2.resize(image, dsize=(800, 700), interpolation=cv2.INTER_CUBIC)
	mpimg.imsave("img" + str(opt) + ".jpeg", image, format='jpg')
	# show the images
	#cv2.imshow("teste", np.hstack([wide, tight, auto]))
	#cv2.imshow("blur + Canny auto", np.hstack([auto]))
	#cv2.waitKey(0)
	mpimg.imsave("img_Blur_canny_" + str(opt) + ".jpeg", auto, format='jpg')
	
	#return(wide, tight, auto)
	return(auto)

def morfologias(img):
	# adaptiveThreshold( source_array, maxValue, Adaptive_Method, Threshold_Type, BlockSize, Constante substracted to the mean)
	th2 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
	th3 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
	
	im_th = th3
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
	plt.show()

	kernel = np.ones((5,5),np.uint8)
	'''
	erosion = cv2.erode(img,kernel,iterations = 1)
	dilation = cv2.dilate(img,kernel,iterations = 1)
	opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel) # erosion followed by dilation
	'''
	# Dilation followed by Erosion. 
	# It is useful in closing small holes inside the foreground objects, or small black points on the object.
	closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

	# difference between dilation and erosion of an image. The result will look like the outline of the object.
	gradient = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel) 
		
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
	plt.show()
	

	# Utilizar a função labeling para remover o fundo interior e ficar só com a borda
	#labels, n = labeling(dil_bin_f_canny)
	#borda_reg = remove_regions(labels)
	

def read_sentinel_images():
	Image.MAX_IMAGE_PIXELS = 1000000000
	dir = "../R10m/"
	print('Reading B04...')
	# [3600:3800, 3500:3700]
	img_red = mpimg.imread(dir + 'T29TNF_20180326T112109_B04_10m.jp2')[3700:3800, 3500:3650]
	#plt.imshow(img_red)
	#plt.show()
	
	print('Reading B03...')
	img_green = mpimg.imread(dir + 'T29TNF_20180326T112109_B03_10m.jp2')[3700:3800, 3500:3650]
	#plt.imshow(img_green)
	#plt.show()
	print('Reading B02...')
	img_blue = mpimg.imread(dir + 'T29TNF_20180326T112109_B02_10m.jp2')[3700:3800, 3500:3650]
	#plt.imshow(img_blue)
	#plt.show()
	
	print("Mounting all together...")
	img = np.dstack((img_red))
	img = np.dstack((img_red, img_green, img_blue))
	max_pixel_value = img.max()
	img = np.multiply(img, 255.0)
	img = np.divide(img, max_pixel_value)
	img = img.astype(np.uint8)
	#mpimg.imsave('img_RGB.jpeg', img, format='jpg')
	return img

def main():
	directory = "../R10m/*.jp2"
	files = glob.glob(directory)
	
	print("Sentinel 2 Spacial Resolution Bands:\n" + 
		"   Band 2 - Blue (490nm)\t\t Band 3 - Green (560nm)\n" + 
		"   Band 4 - Orange (665nm)\t\t Band 5 - Infrared  (842nm)\n")
	
	# Loading sentinel jp2 images and combine then in a proper RGB imagem
	img = read_sentinel_images()
	#print("Apllying canny...")
	#canny_img = canny(img, 0)
	(w, h, c) = img.shape
	img = cv2.resize(img, dsize=(3*h, 3*w), interpolation=cv2.INTER_CUBIC)

	print("Apllying canny...")
	canny_img = canny(img, 1)
	
	morfologias(canny_img)
	'''
	for f in files[1:]: 
		image = cv2.imread(f)
	_, ax = plt.subplots(2,2, figsize = (5,5))
	ax[0,0].imshow(toimage(img1))
	ax[0,0].set_title("img1")
	ax[0,1].imshow(toimage(img2))
	ax[0,1].set_title("img2")
	ax[1,0].imshow(toimage(f_canny1))
	ax[1,1].imshow(toimage(f_canny2))
	plt.show()'''
	
if __name__ == "__main__": main()
