# supporting file for ASL classifiers

import numpy as np
import tensorflow as tf
import math
import glob
import matplotlib.image as mpimg

def random_mini_batches(X,Y, mini_batch_size=64, seed=0):
	
	m            = X.shape[1]
	mini_batches = []
	np.random.seed(seed)
	
	# shuffle (X, Y)
	permutation  = list(np.random.permutation(m))
	shuffled_X   = X[:, permutation]
	shuffled_Y   = Y[:, permutation]
	
	# partition minus the end case
	num_complete_minbatches = math.floor(m/mini_batch_size)
	
	for k in range(0, num_complete_minbatches):
		mini_batch_X = shuffled_X[:,k*mini_batch_size:k*mini_batch_size+mini_batch_size]
		mini_batch_Y = shuffled_Y[:,k*mini_batch_size:k*mini_batch_size+mini_batch_size]
		min_batch    = (mini_batch_X, mini_batch_Y)
		mini_batches.append(min_batch)
	
	if m%mini_batch_size != 0:
		mini_batch_X = shuffled_X[:num_complete_minbatches*mini_batch_size:m]
		mini_batch_Y = shuffled_Y[:num_complete_minbatches*mini_batch_size:m]
		min_batch    = (mini_batch_X, mini_batch_Y)
		mini_batches.append(min_batch)
		
	return mini_batches

def convert_to_one_hot(Y, C):
	Y = np.eye(C)[Y.reshape(-1)].T 
	return Y
	
def predict(X, parameters):
	W1 = parameters['W1']
	b1 = parameters['b1']
	W2 = parameters['W2']
	b2 = parameters['b2']
	W3 = parameters['W3']
	b3 = parameters['b3'] 
														   # Numpy Equivalents:
	Z1 = tf.add(tf.matmul(W1, X), b1)                      # Z1 = np.dot(W1, X) + b1
	A1 = tf.nn.relu(Z1)                                    # A1 = relu(Z1)
	Z2 = tf.add(tf.matmul(W2, A1), b2)                     # Z2 = np.dot(W2, a1) + b2
	A2 = tf.nn.relu(Z2)                                    # A2 = relu(Z2)
	Z3 = tf.add(tf.matmul(W3, A2), b3)                     # Z3 = np.dot(W3,Z2) + b3

	return Z3
	
def data_load(path,C, x_list, y_list):
	folder=glob.glob(path+'/*')
	for imgn in folder:
		img=np.array(mpimg.imread(imgn))
		x_list.append(img)
		y_list.append(C)
	return (x_list, y_list)	
