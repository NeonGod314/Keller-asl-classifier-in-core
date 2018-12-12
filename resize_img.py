# program to resize image for training without loosing resolution
import numpy as np
import cv2
import glob
import os
def img_resize(img, new_size):
	img_resize = cv2.resize(img, (new_size, new_size))
	return img_resize

def img_blur(img):
	img_blur = cv2.blur(img, (3,3))
	return img_blur

def color2grey(img):
	img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	return img_gray

def gray2binary(img):
	_, img_binary = cv2.threshold(img, 210, 255, cv2.THRESH_BINARY)
	return img_binary

def  binary2rgb(img):
	img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
	return img_rgb

image_folder = glob.glob('./del/New folder/*')
for image_file in image_folder:
	img = cv2.imread(image_file)
	#img = color2grey(img)
	size = 64
	filename = os.path.basename(image_file)
	print ("filename: ", filename)
	for i in range(1):
		print ("size: ", size)
		#img = gray2binary(img)
		#img = img_blur(img)
		img = img_resize(img, int(size))
		#img = gray2binary(img)
		size = size/2
	#img = binary2rgb(img)
	print (img.shape)	
	cv2.imwrite('./del/'+str(i)+'_'+filename, img)

