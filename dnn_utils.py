import pickle
import os
import math
import glob
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
# import tensorflow as tf
# from tensorflow.python.framework import ops
	
def load_images(data_directory):
	img_data = []
	img_class= []
	classes  = []
	dataset  = {}

	class_folders = glob.glob('./' + data_directory + '/*')
	
	for specific_class in class_folders:
		class_name=os.path.basename(specific_class)
		classes.append(class_name)

		images = glob.glob('./' + specific_class + '/*')
		for image in images:
			img = Image.open(image)
			img = np.array(img)
			img_data.append(img)
			img_class.append(class_name)

	dict_name_to_label, dict_label_to_name  = convert_className_to_classLabel(classes)
	dataset["set_x"] = img_data	
	dataset["set_y"] = [dict_name_to_label[i] for i in img_class]
	dataset["classes"] = [dict_name_to_label[i] for i in classes]
	
	return len(dataset["classes"]), dict_label_to_name, dataset 

def pickle_generator(dict_label_to_name):
	with open('./modelData/dict_label_to_name.pickle', 'wb') as handle:
		pickle.dump(dict_label_to_name, handle, protocol=pickle.HIGHEST_PROTOCOL)

def load_dataset(train_dataset, test_dataset):
	train_set_x_orig = np.array(train_dataset["set_x"][:]) #train set features
	train_set_y_orig = np.array(train_dataset["set_y"][:]) #train set label

	test_set_x_orig = np.array(test_dataset["set_x"][:]) #test set features
	test_set_y_orig = np.array(test_dataset["set_y"][:]) #test set labels

	classes = np.array(train_dataset["classes"][:]) #list of all classes

	train_set_y_orig = train_set_y_orig.reshape((1,train_set_y_orig.shape[0]))
	test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

	return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes

def convert_to_one_hot(Y, C):
	Y = np.eye(C)[Y.reshape(-1)].T
	return Y

def convert_className_to_classLabel(classes):
	dict_name_to_label = {}
	dict_label_to_name = {}
	
	class_labels = [item[0] for item in enumerate(classes)]

	for i in range(len(class_labels)):
		dict_name_to_label[classes[i]] = class_labels[i]

	for i in range(len(class_labels)):
		dict_label_to_name[class_labels[i]] = classes[i]

	return dict_name_to_label, dict_label_to_name   

def random_mini_batches(X, Y, mini_batch_size = 64, seed = 0):
	m = X.shape[0]
	mini_batches = []
	np.random.seed(seed)

	#step1:shiffle (X,Y)
	permutation = list(np.random.permutation(m))
	shuffled_X  = X[permutation,:,:,:]
	shuffled_Y  = Y[permutation,:]

	#shuffled partition minus end case
	num_complete_minibatches = math.floor(m/mini_batch_size) 

	for k in range(0, num_complete_minibatches):
		mini_batch_X = shuffled_X[k*mini_batch_size:k*mini_batch_size+mini_batch_size,:,:,:]
		mini_batch_Y = shuffled_Y[k*mini_batch_size:k*mini_batch_size+mini_batch_size,:]
		mini_batch   = (mini_batch_X, mini_batch_Y)
		mini_batches.append(mini_batch)

	if m % mini_batch_size != 0:
		mini_batch_X = shuffled_X[num_complete_minibatches*mini_batch_size:m,:,:,:]
		mini_batch_Y = shuffled_Y[num_complete_minibatches*mini_batch_size:m,:]
		mini_batch   = (mini_batch_X, mini_batch_Y)
		mini_batches.append(mini_batch)

	return mini_batches		




