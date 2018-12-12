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

	class_dict = convert_className_to_classLabel(classes)
	dataset["set_x"] = img_data	
	dataset["set_y"] = [class_dict[i] for i in img_class]
	dataset["classes"] = [class_dict[i] for i in classes]
	
	return dataset

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
	class_dict={}
	class_labels = [item[0] for item in enumerate(classes)]

	for i in range(len(class_labels)):
		class_dict[classes[i]]=class_labels[i]

	return class_dict   

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

def load_images_for_inferencing(data_directory):
	img_data = []
	images = glob.glob('./'+data_directory+'/*')

	for image in images:
		img = Image.open(image)
		img = np.array(img)
		img_data.append(img)
	img_data = np.array(img_data)	
	#print (img_data.shape)	
	return img_data

# def inferce_prediction(image_data, parameters):
# 	#X, Y = create_placeholders(n_H0, n_W0, n_C0, n_y)
# 	Z3 = forward_propagation(X, parameters)
# 	predict_op = tf.argmax(Z3, 1)
# 	predicted_label = sess.run
# 	with tf.Session() as sess:
# 		predict_op = tf.argmax(Z3, 1)
# 		correct_prediction = tf.equal(predict_op, tf.argmax(Y, 1))
# 		accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float")).eval({X: image_data})
# 		#test_accuracy = accuracy.eval({X: X_test, Y: Y_test})
		
# 	print ("test accuracy: ", test_accuracy)
# 	return parameters
 
# images_for_inferencing = load_images_for_inferencing('input')	

