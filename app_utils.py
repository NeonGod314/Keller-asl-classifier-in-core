import pickle
import os
import math
import glob
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def load_pickle(pickle_directory):
	dict_label_to_name = pickle.load(open(pickle_directory,"rb"))
	return dict_label_to_name

def load_images_for_inferencing(data_directory):
	file_data    = {}
	input_image = []
	file_loc    = []

	images = glob.glob(data_directory+'/*.jpg')

	for image in images:
		img = Image.open(image)
		img = np.array(img)
		input_image.append(img)
		file_loc.append(image)

	input_image = np.array(input_image)	

	file_data["inferencing_input"] = input_image
	file_data["file_loc"] = file_loc

	return file_data

def convert_classLabel_to_className(label, dict_label_to_name):
	return dict_label_to_name[label]

def create_folders_for_predicted_classes(predicted_labels, dict_label_to_name, input_dir):
	folders = set(predicted_labels)

	for folder in folders:
		try :
			os.mkdir(input_dir + '/' + dict_label_to_name[folder])
			print ("data_directory created : ", input_dir + '/' + dict_label_to_name[folder])
		except FileExistsError:
			print ("Directory already exists : ", input_dir + '/' + dict_label_to_name[folder])

def data_merge(predicted_labels, file_data, dict_label_to_name):
	classNames = []
	
	for i in range(len(predicted_labels)):
		className = dict_label_to_name[predicted_labels[i]]
		classNames.append(className)
	file_data["classNames"] = classNames
	#print (file_data["classNames"])	
	return file_data	

def move_input_to_classFiles(input_dir, file_data):
	for i in range(len(file_data["classNames"])):
		os.rename(file_data["file_loc"][i], input_dir + '/' + file_data['classNames'][i] + '/' + os.path.basename(file_data['file_loc'][i]))