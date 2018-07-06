# Street Segmentation from QGis rasters

## How to interact
 - W: Move up on the raster
-  A: Move left on the raster
-  S: Move down on the raster
-  D: Move right on the raster
- +: Increase the zoom (reduce the scale)
- \-: Decrease the zoom (increase the scale)
- Q: Quit program

## How to run the program
 - Simply run Street_seg.py and two windows will be displayed. 
 - The left one shows the original satellite raster preview
 - The right one displays the results obtained after segmentation. 
 
## Main considerations of the algorithm

- Some visual results are presented at Resultados folder. 

- Only the TCI raster is used since it is an RGB image built from 
the B02 (Blue), B03 (Green), and B04 (Red) Bands. 
Therefore there's no need of reading the images of the 3 individual 
sensors. 

- Because the input is based on a truly HDR image, every time the zoom 
(scale) of the windows are changed, the displayed portion of the raster 
is normalized for an RGB scale between [0,255]. This is done by taking 
into account the maximum value of the pixels of the displayed portion, 
allowing a dynamic adjustment based on the light intensity of that portion. 

- For higher values of zoom, the original raster presents already 
some blur. In these scenarios, applying a gaussian blur filter to reduced soft 
edges before canny is irrelevant and not useful to the final result. 
 
 - An adaptive cauny method is used to detect the edges of the raster. 
 The lower and upper limits are estimated based on the median of the image as seen [here](https://www.pyimagesearch.com/2015/04/06/zero-parameter-automatic-canny-edge-detection-with-python-and-opencv/).
 
  - After applying the canny filters, the green (chlorophyll) zones are 
 "cleaned" from the result. This is made base on an analysis of the pixel 
 colour in the HSV space and allows to remove edges encounter in places like 
 rivers surrounded by green zones or edges between field crops. 


## Team
 - Miguel Miranda

 > Tecnologias e Aplicação de CG - Cartografia
 
 > Mestrado Integrado em Engenharia Informática
 
 > Departamento de Informática
 
 > Universidade do Minho - 2018

