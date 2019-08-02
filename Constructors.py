# ==============================================================================
# Constructor fot the architectures: V7, VGG11, VGG13, VGG16, ResNet 50
# UNAM IIMAS
# Authors: 	Ivette Velez
# 			Caleb Rascon
# 			Gibran Fuentes
#			Alejandro Maldonado
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path
import os
import sys
import time
import glob
import json
import numpy as np
import soundfile as sf
import tensorflow as tf
from sklearn.metrics import roc_curve
from collections import namedtuple
# from tensorflow.contrib import layers

# silences Tensorflow boot logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Using just one GPU in case of GPU 
# os.environ['CUDA_VISIBLE_DEVICES']= '0'

import matplotlib as mpl
if os.environ.get('DISPLAY','') == '':
		print('no display found. Using non-interactive Agg backend')
		mpl.use('Agg')
import matplotlib.pyplot as plt

class Model():

	def __init__(self, 
		in_height,
		in_width,
		channels,
		label,
		pool_layers):

		self.in_height = in_height
		self.in_width = in_width
		self.channels = channels
		self.label = label
		self.width_after_conv = int(np.ceil(float(self.in_width)/float(2**pool_layers)))
		self.height_after_conv = int(np.ceil(float(self.in_height)/float(2**pool_layers)))

		# To avoid future errors initializing all the variables
		self.X1 = None
		self.X2 = None
		self.Y = None
		self.training = None
		self.g_step = None
		self.Y_pred = None
		self.label_pred = None
		self.label_true = None
		self.accuracy = None
		self.acc_batch = None
		self.loss = None
		self.Y_logt = None

		""" Creates the model """
		self.def_input()
		self.def_params()
		self.def_model()
		self.def_output()
		self.def_loss()
		self.def_metrics()
		self.add_summaries()

	def conv2d(self, x, W):
		"""conv2d returns a 2d convolution layer with full stride."""
		return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

	def max_pool_2x2(self, x):
		"""max_pool_2x2 downsamples a feature map by 2X."""
		return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
								strides=[1, 2, 2, 1], padding='SAME')

	def weight_variable(self,shape):
		initializer = tf.contrib.layers.xavier_initializer()(shape)
		return tf.Variable(initializer)
		
	def bias_variable(self,shape):
		initializer = tf.contrib.layers.xavier_initializer()(shape)
		return tf.Variable(initializer)

	def def_input(self):
		""" Defines inputs """
		with tf.name_scope('input'):

			# Defining the entrance of the model
			self.X1 = tf.placeholder(tf.float32, [None, self.in_height, self.in_width, self.channels], name='X1')
			self.X2 = tf.placeholder(tf.float32, [None, self.in_height, self.in_width, self.channels], name='X2')
			self.Y = tf.placeholder(tf.float32, [None, self.label], name='Y')

			self.training = tf.placeholder(tf.bool, name='training')
			self.g_step = tf.contrib.framework.get_or_create_global_step()

	def def_model(self):
		""" Defines the model """
		self.Y_logt = tf.constant(0, shape=[self.label]);		

	def def_output(self):
		""" Defines model output """
		with tf.name_scope('output'):
			self.Y_pred = tf.nn.softmax(self.Y_logt, name='Y_pred')
			self.label_pred = tf.argmax(self.Y_pred, 1, name='label_pred')
			self.label_true = tf.argmax(self.Y, 1, name='label_true')

	def def_metrics(self):
		""" Adds metrics """
		with tf.name_scope('metrics'):
			cmp_labels = tf.equal(self.label_true, self.label_pred)
			self.accuracy = tf.reduce_sum(tf.cast(cmp_labels, tf.float32), name='accuracy')
			self.acc_batch = tf.reduce_mean(tf.cast(cmp_labels, tf.float32))*100

	def add_summaries(self):
		""" Adds summaries for Tensorboard """
		with tf.name_scope('summaries'):
			tf.summary.scalar('loss', self.loss)
			tf.summary.scalar('accuracy', self.acc_batch)
			self.summary = tf.summary.merge_all()

	def def_loss(self):
		""" Defines loss function """
		self.loss = tf.constant(0);

	def def_params(self):
		""" Defines model parameters """		


class V7(Model):

	def __init_(self, 
		in_height,
		in_width,
		channels,
		label,
		pool_layers):
		Model.__init__(in_height,in_width,channels,label,pool_layers)

	def def_loss(self):
		""" Defines loss function """
		with tf.name_scope('loss'):
			#cross entropy
			self.cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=self.Y, logits=self.Y_logt)
			self.loss = tf.reduce_mean(self.cross_entropy)	

	def def_params(self):
		""" Defines model parameters """
		with tf.name_scope('params'):

			# First convolutional layer
			with tf.name_scope('conv1'):
				self.W_cn1 = self.weight_variable([3, 3, 1, 64])
				self.b_cn1 = self.bias_variable([64])

			# Second convolutional layer
			with tf.name_scope('conv2'):
				self.W_cn2 = self.weight_variable([3, 3, 64, 64])
				self.b_cn2 = self.bias_variable([64])

			# Third convolutional layer
			with tf.name_scope('conv3'):
				self.W_cn3 = self.weight_variable([3, 3, 64, 128])
				self.b_cn3 = self.bias_variable([128])

			# Fourth Convolutional layer
			with tf.name_scope('conv4'):
				self.W_cn4 = self.weight_variable([3, 3, 128, 128])
				self.b_cn4 = self.bias_variable([128])

			# First fully connected layer			
			with tf.name_scope('fc1'):
				self.W_fc1 = self.weight_variable([self.height_after_conv * self.width_after_conv * 256, 1024])
				self.b_fc1 = self.bias_variable([1024])

			# Second fully connected layer			
			with tf.name_scope('fc2'):
				self.W_fc2 = self.weight_variable([1024, 1024])
				self.b_fc2 = self.bias_variable([1024])

			# Third fully connected layer
			with tf.name_scope('fc3'):
				self.W_fc3 = self.weight_variable([1024, self.label])
				self.b_fc3 = self.bias_variable([self.label])

	def def_model(self):
		""" Defines the model """
		W_cn1 = self.W_cn1
		b_cn1 = self.b_cn1
		W_cn2 = self.W_cn2
		b_cn2 = self.b_cn2
		W_cn3 = self.W_cn3
		b_cn3 = self.b_cn3
		W_cn4 = self.W_cn4
		b_cn4 = self.b_cn4
		W_fc1 = self.W_fc1
		b_fc1 = self.b_fc1
		W_fc2 = self.W_fc2
		b_fc2 = self.b_fc2
		W_fc3 = self.W_fc3
		b_fc3 = self.b_fc3
	
		# First convolutional layers for the first signal
		with tf.name_scope('conv1a'):
			h_cn1a = tf.nn.relu(self.conv2d(self.X1, W_cn1) + b_cn1)

		# First convolutional layers for the second signal
		with tf.name_scope('conv1b'):
			h_cn1b = tf.nn.relu(self.conv2d(self.X2, W_cn1) + b_cn1)

		# Second convolutional layers for the first signal
		with tf.name_scope('conv2a'):
			h_cn2a = tf.nn.relu(self.conv2d(h_cn1a, W_cn2) + b_cn2)

		# Second convolutional layers for the second signal
		with tf.name_scope('conv2b'):
			h_cn2b = tf.nn.relu(self.conv2d(h_cn1b, W_cn2) + b_cn2)

		# First pooling layer for the first signal
		with tf.name_scope('pool1a'):
			h_pool1a = self.max_pool_2x2(h_cn2a)

		# First pooling layer for the second signal
		with tf.name_scope('pool1b'):
			h_pool1b = self.max_pool_2x2(h_cn2b)

		# Third convolutional layers for the first signal
		with tf.name_scope('conv3a'):
			h_cn3a = tf.nn.relu(self.conv2d(h_pool1a, W_cn3) + b_cn3)

		# Third convolutional layers for the second signal
		with tf.name_scope('conv3b'):
			h_cn3b = tf.nn.relu(self.conv2d(h_pool1b, W_cn3) + b_cn3)

		# Fourth convolutional layers for the first signal
		with tf.name_scope('conv4a'):
			h_cn4a = tf.nn.relu(self.conv2d(h_cn3a, W_cn4) + b_cn4)

		# Fourth convolutional layers for the second signal
		with tf.name_scope('conv4b'):
			h_cn4b = tf.nn.relu(self.conv2d(h_cn3b, W_cn4) + b_cn4)

		# Second pooling layer for the first signal
		with tf.name_scope('pool2a'):
			h_pool2a = self.max_pool_2x2(h_cn4a)

		# Second pooling layer for the second signal
		with tf.name_scope('pool2b'):
			h_pool2b = self.max_pool_2x2(h_cn4b)

		# Concat layer to go from convolutional layer to fully connected
		with tf.name_scope('concat1'):
			h_concat1 = tf.concat([h_pool2a, h_pool2b], axis=3)

		# First fully connected layer
		with tf.name_scope('fc1'):
			h_concat1_flat = tf.reshape(h_concat1, [-1, self.height_after_conv * self.width_after_conv *256]) 
			h_mat =	tf.matmul(h_concat1_flat, W_fc1)
			h_fc1 = tf.nn.relu(h_mat + b_fc1)

		# Second fully connected layer
		with tf.name_scope('fc2'):
			h_fc2 = tf.nn.relu(tf.matmul(h_fc1, W_fc2) + b_fc2)

		# Third fully connected layer
		with tf.name_scope('fc3'):
			self.Y_logt = tf.matmul(h_fc2, W_fc3) + b_fc3


class VGG11(Model):
	def __init_(self, 
		in_height,
		in_width,
		channels,
		label,
		pool_layers):
		Model.__init__(in_height,in_width,channels,label,pool_layers)

	def def_loss(self):
		""" Defines loss function """
		with tf.name_scope('loss'):
			#cross entropy
			self.cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=self.Y, logits=self.Y_logt)
			self.loss = tf.reduce_mean(self.cross_entropy)

	def def_params(self):
		""" Defines model parameters """
		with tf.name_scope('params'):

			# First convolutional layer
			with tf.name_scope('conv1'):
				self.W_cn1 = self.weight_variable([3, 3, 1, 64])
				self.b_cn1 = self.bias_variable([64])

			# Second convolutional layer
			with tf.name_scope('conv2'):
				self.W_cn2 = self.weight_variable([3, 3, 64, 128])
				self.b_cn2 = self.bias_variable([128])

			# Third convolutional layer
			with tf.name_scope('conv3'):
				self.W_cn3 = self.weight_variable([3, 3, 128, 256])
				self.b_cn3 = self.bias_variable([256])

			# Fourth Convolutional layer
			with tf.name_scope('conv4'):
				self.W_cn4 = self.weight_variable([3, 3, 256, 256])
				self.b_cn4 = self.bias_variable([256])

			# Fifth Convolutional layer
			with tf.name_scope('conv5'):
				self.W_cn5 = self.weight_variable([3, 3, 256, 512])
				self.b_cn5 = self.bias_variable([512])

			# Sixth Convolutional layer
			with tf.name_scope('conv6'):
				self.W_cn6 = self.weight_variable([3, 3, 512, 512])
				self.b_cn6 = self.bias_variable([512])

			# Seventh Convolutional layer
			with tf.name_scope('conv7'):
				self.W_cn7 = self.weight_variable([3, 3, 512, 512])
				self.b_cn7 = self.bias_variable([512])

			# Eighth Convolutional layer
			with tf.name_scope('conv8'):
				self.W_cn8 = self.weight_variable([3, 3, 512, 512])
				self.b_cn8 = self.bias_variable([512])


			# First fully connected layer			
			with tf.name_scope('fc1'):
				self.W_fc1 = self.weight_variable([self.height_after_conv * self.width_after_conv * 512*2, 1024])
				self.b_fc1 = self.bias_variable([1024])

			# Second fully connected layer			
			with tf.name_scope('fc2'):
				self.W_fc2 = self.weight_variable([1024, 1024])
				self.b_fc2 = self.bias_variable([1024])

			# Third fully connected layer
			with tf.name_scope('fc3'):
				self.W_fc3 = self.weight_variable([1024, self.label])
				self.b_fc3 = self.bias_variable([self.label])				

	def def_model(self):
		""" Defines the model """
		W_cn1 = self.W_cn1
		b_cn1 = self.b_cn1
		W_cn2 = self.W_cn2
		b_cn2 = self.b_cn2
		W_cn3 = self.W_cn3
		b_cn3 = self.b_cn3
		W_cn4 = self.W_cn4
		b_cn4 = self.b_cn4
		W_cn5 = self.W_cn5
		b_cn5 = self.b_cn5
		W_cn6 = self.W_cn6
		b_cn6 = self.b_cn6
		W_cn7 = self.W_cn7
		b_cn7 = self.b_cn7
		W_cn8 = self.W_cn8
		b_cn8 = self.b_cn8
		W_fc1 = self.W_fc1
		b_fc1 = self.b_fc1
		W_fc2 = self.W_fc2
		b_fc2 = self.b_fc2
		W_fc3 = self.W_fc3
		b_fc3 = self.b_fc3
	
		# First convolutional layers for the first signal
		with tf.name_scope('conv1a'):
			h_cn1a = tf.nn.relu(self.conv2d(self.X1, W_cn1) + b_cn1)

		# First convolutional layers for the second signal
		with tf.name_scope('conv1b'):
			h_cn1b = tf.nn.relu(self.conv2d(self.X2, W_cn1) + b_cn1)

		# First pooling layer for the first signal
		with tf.name_scope('pool1a'):
			h_pool1a = self.max_pool_2x2(h_cn1a)

		# First pooling layer for the second signal
		with tf.name_scope('pool1b'):
			h_pool1b = self.max_pool_2x2(h_cn1b)

		# Second convolutional layers for the first signal
		with tf.name_scope('conv2a'):
			h_cn2a = tf.nn.relu(self.conv2d(h_pool1a, W_cn2) + b_cn2)

		# Second convolutional layers for the second signal
		with tf.name_scope('conv2b'):
			h_cn2b = tf.nn.relu(self.conv2d(h_pool1b, W_cn2) + b_cn2)

		# Second pooling layer for the first signal
		with tf.name_scope('pool2a'):
			h_pool2a = self.max_pool_2x2(h_cn2a)

		# Second pooling layer for the second signal
		with tf.name_scope('pool2b'):
			h_pool2b = self.max_pool_2x2(h_cn2b)

		# Third convolutional layers for the first signal
		with tf.name_scope('conv3a'):
			h_cn3a = tf.nn.relu(self.conv2d(h_pool2a, W_cn3) + b_cn3)

		# Third convolutional layers for the second signal
		with tf.name_scope('conv3b'):
			h_cn3b = tf.nn.relu(self.conv2d(h_pool2b, W_cn3) + b_cn3)

		# Fourth convolutional layers for the first signal
		with tf.name_scope('conv4a'):
			h_cn4a = tf.nn.relu(self.conv2d(h_cn3a, W_cn4) + b_cn4)

		# Fourth convolutional layers for the second signal
		with tf.name_scope('conv4b'):
			h_cn4b = tf.nn.relu(self.conv2d(h_cn3b, W_cn4) + b_cn4)

		# Third pooling layer for the first signal
		with tf.name_scope('pool3a'):
			h_pool3a = self.max_pool_2x2(h_cn4a)

		# Second pooling layer for the second signal
		with tf.name_scope('pool3b'):
			h_pool3b = self.max_pool_2x2(h_cn4b)

		# Fifth convolutional layers for the first signal
		with tf.name_scope('conv5a'):
			h_cn5a = tf.nn.relu(self.conv2d(h_pool3a, W_cn5) + b_cn5)

		# Fifth convolutional layers for the second signal
		with tf.name_scope('conv5b'):
			h_cn5b = tf.nn.relu(self.conv2d(h_pool3b, W_cn5) + b_cn5)

		# Sixth convolutional layers for the first signal
		with tf.name_scope('conv6a'):
			h_cn6a = tf.nn.relu(self.conv2d(h_cn5a, W_cn6) + b_cn6)

		# Sixth convolutional layers for the second signal
		with tf.name_scope('conv6b'):
			h_cn6b = tf.nn.relu(self.conv2d(h_cn5b, W_cn6) + b_cn6)

		# Fourth pooling layer for the first signal
		with tf.name_scope('pool4a'):
			h_pool4a = self.max_pool_2x2(h_cn6a)

		# Second pooling layer for the second signal
		with tf.name_scope('pool4b'):
			h_pool4b = self.max_pool_2x2(h_cn6b)

		# Seventh convolutional layers for the first signal
		with tf.name_scope('conv7a'):
			h_cn7a = tf.nn.relu(self.conv2d(h_pool4a, W_cn7) + b_cn7)

		# Seventh convolutional layers for the second signal
		with tf.name_scope('conv7b'):
			h_cn7b = tf.nn.relu(self.conv2d(h_pool4b, W_cn7) + b_cn7)

		# Seventh convolutional layers for the first signal
		with tf.name_scope('conv8a'):
			h_cn8a = tf.nn.relu(self.conv2d(h_cn7a, W_cn8) + b_cn8)

		# Seventh convolutional layers for the second signal
		with tf.name_scope('conv8b'):
			h_cn8b = tf.nn.relu(self.conv2d(h_cn7b, W_cn8) + b_cn8)

		# Fourth pooling layer for the first signal
		with tf.name_scope('pool5a'):
			h_pool5a = self.max_pool_2x2(h_cn8a)

		# Second pooling layer for the second signal
		with tf.name_scope('pool5b'):
			h_pool5b = self.max_pool_2x2(h_cn8b)

		# Concat layer to go from convolutional layer to fully connected
		with tf.name_scope('concat1'):
			h_concat1 = tf.concat([h_pool5a, h_pool5b], axis=3)

		# First fully connected layer
		with tf.name_scope('fc1'):
			h_concat1_flat = tf.reshape(h_concat1, [-1, self.height_after_conv * self.width_after_conv *512*2]) 
			h_mat =	tf.matmul(h_concat1_flat, W_fc1)
			h_fc1 = tf.nn.relu(h_mat + b_fc1)

		# Second fully connected layer
		with tf.name_scope('fc2'):
			h_fc2 = tf.nn.relu(tf.matmul(h_fc1, W_fc2) + b_fc2)

		# Third fully connected layer
		with tf.name_scope('fc3'):
			self.Y_logt = tf.matmul(h_fc2, W_fc3) + b_fc3


class VGG13(Model):
	def __init_(self, 
		in_height,
		in_width,
		channels,
		label,
		pool_layers):
		Model.__init__(in_height,in_width,channels,label,pool_layers)

	def def_loss(self):
		""" Defines loss function """
		with tf.name_scope('loss'):
			#cross entropy
			self.cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=self.Y, logits=self.Y_logt)
			self.loss = tf.reduce_mean(self.cross_entropy)

	def def_params(self):
		""" Defines model parameters """
		with tf.name_scope('params'):

			# First convolutional layer
			with tf.name_scope('conv1'):
				self.W_cn1 = self.weight_variable([3, 3, 1, 64])
				self.b_cn1 = self.bias_variable([64])

			# Second convolutional layer
			with tf.name_scope('conv2'):
				self.W_cn2 = self.weight_variable([3, 3, 64, 64])
				self.b_cn2 = self.bias_variable([64])

			# Third convolutional layer
			with tf.name_scope('conv3'):
				self.W_cn3 = self.weight_variable([3, 3, 64, 128])
				self.b_cn3 = self.bias_variable([128])

			# Fourth Convolutional layer
			with tf.name_scope('conv4'):
				self.W_cn4 = self.weight_variable([3, 3, 128, 128])
				self.b_cn4 = self.bias_variable([128])

			# Fifth Convolutional layer
			with tf.name_scope('conv5'):
				self.W_cn5 = self.weight_variable([3, 3, 128, 256])
				self.b_cn5 = self.bias_variable([256])

			# Sixth Convolutional layer
			with tf.name_scope('conv6'):
				self.W_cn6 = self.weight_variable([3, 3, 256, 256])
				self.b_cn6 = self.bias_variable([256])

			# Seventh Convolutional layer
			with tf.name_scope('conv7'):
				self.W_cn7 = self.weight_variable([3, 3, 256, 512])
				self.b_cn7 = self.bias_variable([512])

			# Eighth Convolutional layer
			with tf.name_scope('conv8'):
				self.W_cn8 = self.weight_variable([3, 3, 512, 512])
				self.b_cn8 = self.bias_variable([512])

			# Nineth Convolutional layer
			with tf.name_scope('conv9'):
				self.W_cn9 = self.weight_variable([3, 3, 512, 512])
				self.b_cn9 = self.bias_variable([512])

			# Tenth Convolutional layer
			with tf.name_scope('conv10'):
				self.W_cn10 = self.weight_variable([3, 3, 512, 512])
				self.b_cn10 = self.bias_variable([512])

			# First fully connected layer			
			with tf.name_scope('fc1'):
				self.W_fc1 = self.weight_variable([self.height_after_conv * self.width_after_conv * 512*2, 1024])
				self.b_fc1 = self.bias_variable([1024])

			# Second fully connected layer			
			with tf.name_scope('fc2'):
				self.W_fc2 = self.weight_variable([1024, 1024])
				self.b_fc2 = self.bias_variable([1024])

			# Third fully connected layer
			with tf.name_scope('fc3'):
				self.W_fc3 = self.weight_variable([1024, self.label])
				self.b_fc3 = self.bias_variable([self.label])				

	def def_model(self):
		""" Defines the model """
		W_cn1 = self.W_cn1
		b_cn1 = self.b_cn1
		W_cn2 = self.W_cn2
		b_cn2 = self.b_cn2
		W_cn3 = self.W_cn3
		b_cn3 = self.b_cn3
		W_cn4 = self.W_cn4
		b_cn4 = self.b_cn4
		W_cn5 = self.W_cn5
		b_cn5 = self.b_cn5
		W_cn6 = self.W_cn6
		b_cn6 = self.b_cn6
		W_cn7 = self.W_cn7
		b_cn7 = self.b_cn7
		W_cn8 = self.W_cn8
		b_cn8 = self.b_cn8
		W_cn9 = self.W_cn9
		b_cn9 = self.b_cn9
		W_cn10 = self.W_cn10
		b_cn10 = self.b_cn10
		W_fc1 = self.W_fc1
		b_fc1 = self.b_fc1
		W_fc2 = self.W_fc2
		b_fc2 = self.b_fc2
		W_fc3 = self.W_fc3
		b_fc3 = self.b_fc3
	
		# First convolutional layers for the first signal
		with tf.name_scope('conv1a'):
			h_cn1a = tf.nn.relu(self.conv2d(self.X1, W_cn1) + b_cn1)

		# First convolutional layers for the second signal
		with tf.name_scope('conv1b'):
			h_cn1b = tf.nn.relu(self.conv2d(self.X2, W_cn1) + b_cn1)

		# Second convolutional layers for the first signal
		with tf.name_scope('conv2a'):
			h_cn2a = tf.nn.relu(self.conv2d(h_cn1a, W_cn2) + b_cn2)

		# Second convolutional layers for the second signal
		with tf.name_scope('conv2b'):
			h_cn2b = tf.nn.relu(self.conv2d(h_cn1b, W_cn2) + b_cn2)

		# First pooling layer for the first signal
		with tf.name_scope('pool1a'):
			h_pool1a = self.max_pool_2x2(h_cn2a)

		# First pooling layer for the second signal
		with tf.name_scope('pool1b'):
			h_pool1b = self.max_pool_2x2(h_cn2b)

		# Third convolutional layers for the first signal
		with tf.name_scope('conv3a'):
			h_cn3a = tf.nn.relu(self.conv2d(h_pool1a, W_cn3) + b_cn3)

		# Third convolutional layers for the second signal
		with tf.name_scope('conv3b'):
			h_cn3b = tf.nn.relu(self.conv2d(h_pool1b, W_cn3) + b_cn3)

		# Fourth convolutional layers for the first signal
		with tf.name_scope('conv4a'):
			h_cn4a = tf.nn.relu(self.conv2d(h_cn3a, W_cn4) + b_cn4)

		# Fourth convolutional layers for the second signal
		with tf.name_scope('conv4b'):
			h_cn4b = tf.nn.relu(self.conv2d(h_cn3b, W_cn4) + b_cn4)

		# Second pooling layer for the first signal
		with tf.name_scope('pool2a'):
			h_pool2a = self.max_pool_2x2(h_cn4a)

		# Second pooling layer for the second signal
		with tf.name_scope('pool2b'):
			h_pool2b = self.max_pool_2x2(h_cn4b)

		# Fifth convolutional layers for the first signal
		with tf.name_scope('conv5a'):
			h_cn5a = tf.nn.relu(self.conv2d(h_pool2a, W_cn5) + b_cn5)

		# Fifth convolutional layers for the second signal
		with tf.name_scope('conv5b'):
			h_cn5b = tf.nn.relu(self.conv2d(h_pool2b, W_cn5) + b_cn5)

		# Sixth convolutional layers for the first signal
		with tf.name_scope('conv6a'):
			h_cn6a = tf.nn.relu(self.conv2d(h_cn5a, W_cn6) + b_cn6)

		# Sixth convolutional layers for the second signal
		with tf.name_scope('conv6b'):
			h_cn6b = tf.nn.relu(self.conv2d(h_cn5b, W_cn6) + b_cn6)

		# Third pooling layer for the first signal
		with tf.name_scope('pool3a'):
			h_pool3a = self.max_pool_2x2(h_cn6a)

		# Second pooling layer for the second signal
		with tf.name_scope('pool3b'):
			h_pool3b = self.max_pool_2x2(h_cn6b)

		# Seventh convolutional layers for the first signal
		with tf.name_scope('conv7a'):
			h_cn7a = tf.nn.relu(self.conv2d(h_pool3a, W_cn7) + b_cn7)

		# Seventh convolutional layers for the second signal
		with tf.name_scope('conv7b'):
			h_cn7b = tf.nn.relu(self.conv2d(h_pool3b, W_cn7) + b_cn7)

		# Eighth convolutional layers for the first signal
		with tf.name_scope('conv8a'):
			h_cn8a = tf.nn.relu(self.conv2d(h_cn7a, W_cn8) + b_cn8)

		# Eighth convolutional layers for the second signal
		with tf.name_scope('conv8b'):
			h_cn8b = tf.nn.relu(self.conv2d(h_cn7b, W_cn8) + b_cn8)

		# Fourth pooling layer for the first signal
		with tf.name_scope('pool4a'):
			h_pool4a = self.max_pool_2x2(h_cn8a)

		# Fourth pooling layer for the second signal
		with tf.name_scope('pool4b'):
			h_pool4b = self.max_pool_2x2(h_cn8b)

		# Nineth convolutional layers for the first signal
		with tf.name_scope('conv9a'):
			h_cn9a = tf.nn.relu(self.conv2d(h_pool4a, W_cn9) + b_cn9)

		# Nineth convolutional layers for the second signal
		with tf.name_scope('conv9b'):
			h_cn9b = tf.nn.relu(self.conv2d(h_pool4b, W_cn9) + b_cn9)

		# Tenth convolutional layers for the first signal
		with tf.name_scope('conv10a'):
			h_cn10a = tf.nn.relu(self.conv2d(h_cn9a, W_cn10) + b_cn10)

		# Tenth convolutional layers for the second signal
		with tf.name_scope('conv10b'):
			h_cn10b = tf.nn.relu(self.conv2d(h_cn9b, W_cn10) + b_cn10)

		# Fourth pooling layer for the first signal
		with tf.name_scope('pool5a'):
			h_pool5a = self.max_pool_2x2(h_cn10a)

		# Second pooling layer for the second signal
		with tf.name_scope('pool5b'):
			h_pool5b = self.max_pool_2x2(h_cn10b)

		# Concat layer to go from convolutional layer to fully connected
		with tf.name_scope('concat1'):
			h_concat1 = tf.concat([h_pool5a, h_pool5b], axis=3)

		# First fully connected layer
		with tf.name_scope('fc1'):
			h_concat1_flat = tf.reshape(h_concat1, [-1, self.height_after_conv * self.width_after_conv *512*2]) 
			h_mat =	tf.matmul(h_concat1_flat, W_fc1)
			h_fc1 = tf.nn.relu(h_mat + b_fc1)

		# Second fully connected layer
		with tf.name_scope('fc2'):
			h_fc2 = tf.nn.relu(tf.matmul(h_fc1, W_fc2) + b_fc2)

		# Third fully connected layer
		with tf.name_scope('fc3'):
			self.Y_logt = tf.matmul(h_fc2, W_fc3) + b_fc3


class VGG16(Model):
	def __init_(self, 
		in_height,
		in_width,
		channels,
		label,
		pool_layers):
		Model.__init__(in_height,in_width,channels,label,pool_layers)

	def def_loss(self):
		""" Defines loss function """
		with tf.name_scope('loss'):
			#cross entropy
			self.cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=self.Y, logits=self.Y_logt)
			self.loss = tf.reduce_mean(self.cross_entropy)

	def def_params(self):
		""" Defines model parameters """
		with tf.name_scope('params'):

			# First convolutional layer
			with tf.name_scope('conv1'):
				self.W_cn1 = self.weight_variable([3, 3, 1, 64])
				self.b_cn1 = self.bias_variable([64])

			# Second convolutional layer
			with tf.name_scope('conv2'):
				self.W_cn2 = self.weight_variable([3, 3, 64, 64])
				self.b_cn2 = self.bias_variable([64])

			# Third convolutional layer
			with tf.name_scope('conv3'):
				self.W_cn3 = self.weight_variable([3, 3, 64, 128])
				self.b_cn3 = self.bias_variable([128])

			# Fourth Convolutional layer
			with tf.name_scope('conv4'):
				self.W_cn4 = self.weight_variable([3, 3, 128, 128])
				self.b_cn4 = self.bias_variable([128])

			# Fifth Convolutional layer
			with tf.name_scope('conv5'):
				self.W_cn5 = self.weight_variable([3, 3, 128, 256])
				self.b_cn5 = self.bias_variable([256])

			# Sixth Convolutional layer
			with tf.name_scope('conv6'):
				self.W_cn6 = self.weight_variable([3, 3, 256, 256])
				self.b_cn6 = self.bias_variable([256])

			# Seventh Convolutional layer
			with tf.name_scope('conv7'):
				self.W_cn7 = self.weight_variable([3, 3, 256, 256])
				self.b_cn7 = self.bias_variable([256])

			# Eighth Convolutional layer
			with tf.name_scope('conv8'):
				self.W_cn8 = self.weight_variable([3, 3, 256, 512])
				self.b_cn8 = self.bias_variable([512])

			# Nineth Convolutional layer
			with tf.name_scope('conv9'):
				self.W_cn9 = self.weight_variable([3, 3, 512, 512])
				self.b_cn9 = self.bias_variable([512])

			# Tenth Convolutional layer
			with tf.name_scope('conv10'):
				self.W_cn10 = self.weight_variable([3, 3, 512, 512])
				self.b_cn10 = self.bias_variable([512])

			# Eleventh Convolutional layer
			with tf.name_scope('conv11'):
				self.W_cn11 = self.weight_variable([3, 3, 512, 512])
				self.b_cn11 = self.bias_variable([512])

			# Twelveth Convolutional layer
			with tf.name_scope('conv12'):
				self.W_cn12 = self.weight_variable([3, 3, 512, 512])
				self.b_cn12 = self.bias_variable([512])

			# Thirteenth Convolutional layer
			with tf.name_scope('conv13'):
				self.W_cn13 = self.weight_variable([3, 3, 512, 512])
				self.b_cn13 = self.bias_variable([512])

			# First fully connected layer			
			with tf.name_scope('fc1'):
				self.W_fc1 = self.weight_variable([self.height_after_conv * self.width_after_conv * 512*2, 1024])
				self.b_fc1 = self.bias_variable([1024])

			# Second fully connected layer			
			with tf.name_scope('fc2'):
				self.W_fc2 = self.weight_variable([1024, 1024])
				self.b_fc2 = self.bias_variable([1024])

			# Third fully connected layer
			with tf.name_scope('fc3'):
				self.W_fc3 = self.weight_variable([1024, self.label])
				self.b_fc3 = self.bias_variable([self.label])				 

	def def_model(self):
		""" Defines the model """
		W_cn1 = self.W_cn1
		b_cn1 = self.b_cn1
		W_cn2 = self.W_cn2
		b_cn2 = self.b_cn2
		W_cn3 = self.W_cn3
		b_cn3 = self.b_cn3
		W_cn4 = self.W_cn4
		b_cn4 = self.b_cn4
		W_cn5 = self.W_cn5
		b_cn5 = self.b_cn5
		W_cn6 = self.W_cn6
		b_cn6 = self.b_cn6
		W_cn7 = self.W_cn7
		b_cn7 = self.b_cn7
		W_cn8 = self.W_cn8
		b_cn8 = self.b_cn8
		W_cn9 = self.W_cn9
		b_cn9 = self.b_cn9
		W_cn10 = self.W_cn10
		b_cn10 = self.b_cn10
		W_cn11 = self.W_cn11
		b_cn11 = self.b_cn11
		W_cn12 = self.W_cn12
		b_cn12 = self.b_cn12
		W_cn13 = self.W_cn13
		b_cn13 = self.b_cn13
		W_fc1 = self.W_fc1
		b_fc1 = self.b_fc1
		W_fc2 = self.W_fc2
		b_fc2 = self.b_fc2
		W_fc3 = self.W_fc3
		b_fc3 = self.b_fc3
	
		# First convolutional layers for the first signal
		with tf.name_scope('conv1a'):
			h_cn1a = tf.nn.relu(self.conv2d(self.X1, W_cn1) + b_cn1)

		# First convolutional layers for the second signal
		with tf.name_scope('conv1b'):
			h_cn1b = tf.nn.relu(self.conv2d(self.X2, W_cn1) + b_cn1)

		# Second convolutional layers for the first signal
		with tf.name_scope('conv2a'):
			h_cn2a = tf.nn.relu(self.conv2d(h_cn1a, W_cn2) + b_cn2)

		# Second convolutional layers for the second signal
		with tf.name_scope('conv2b'):
			h_cn2b = tf.nn.relu(self.conv2d(h_cn1b, W_cn2) + b_cn2)

		# First pooling layer for the first signal
		with tf.name_scope('pool1a'):
			h_pool1a = self.max_pool_2x2(h_cn2a)

		# First pooling layer for the second signal
		with tf.name_scope('pool1b'):
			h_pool1b = self.max_pool_2x2(h_cn2b)

		# Third convolutional layers for the first signal
		with tf.name_scope('conv3a'):
			h_cn3a = tf.nn.relu(self.conv2d(h_pool1a, W_cn3) + b_cn3)

		# Third convolutional layers for the second signal
		with tf.name_scope('conv3b'):
			h_cn3b = tf.nn.relu(self.conv2d(h_pool1b, W_cn3) + b_cn3)

		# Fourth convolutional layers for the first signal
		with tf.name_scope('conv4a'):
			h_cn4a = tf.nn.relu(self.conv2d(h_cn3a, W_cn4) + b_cn4)

		# Fourth convolutional layers for the second signal
		with tf.name_scope('conv4b'):
			h_cn4b = tf.nn.relu(self.conv2d(h_cn3b, W_cn4) + b_cn4)

		# Second pooling layer for the first signal
		with tf.name_scope('pool2a'):
			h_pool2a = self.max_pool_2x2(h_cn4a)

		# Second pooling layer for the second signal
		with tf.name_scope('pool2b'):
			h_pool2b = self.max_pool_2x2(h_cn4b)

		# Fifth convolutional layers for the first signal
		with tf.name_scope('conv5a'):
			h_cn5a = tf.nn.relu(self.conv2d(h_pool2a, W_cn5) + b_cn5)

		# Fifth convolutional layers for the second signal
		with tf.name_scope('conv5b'):
			h_cn5b = tf.nn.relu(self.conv2d(h_pool2b, W_cn5) + b_cn5)

		# Sixth convolutional layers for the first signal
		with tf.name_scope('conv6a'):
			h_cn6a = tf.nn.relu(self.conv2d(h_cn5a, W_cn6) + b_cn6)

		# Sixth convolutional layers for the second signal
		with tf.name_scope('conv6b'):
			h_cn6b = tf.nn.relu(self.conv2d(h_cn5b, W_cn6) + b_cn6)

		# Seventh convolutional layers for the first signal
		with tf.name_scope('conv7a'):
			h_cn7a = tf.nn.relu(self.conv2d(h_cn6a, W_cn7) + b_cn7)

		# Seventh convolutional layers for the second signal
		with tf.name_scope('conv7b'):
			h_cn7b = tf.nn.relu(self.conv2d(h_cn6b, W_cn7) + b_cn7)

		# Third pooling layer for the first signal
		with tf.name_scope('pool3a'):
			h_pool3a = self.max_pool_2x2(h_cn7a)

		# Second pooling layer for the second signal
		with tf.name_scope('pool3b'):
			h_pool3b = self.max_pool_2x2(h_cn7b)

		# Eighth convolutional layers for the first signal
		with tf.name_scope('conv8a'):
			h_cn8a = tf.nn.relu(self.conv2d(h_pool3a, W_cn8) + b_cn8)

		# Eighth convolutional layers for the second signal
		with tf.name_scope('conv8b'):
			h_cn8b = tf.nn.relu(self.conv2d(h_pool3b, W_cn8) + b_cn8)

		# Nineth convolutional layers for the first signal
		with tf.name_scope('conv9a'):
			h_cn9a = tf.nn.relu(self.conv2d(h_cn8a, W_cn9) + b_cn9)

		# Nineth convolutional layers for the second signal
		with tf.name_scope('conv9b'):
			h_cn9b = tf.nn.relu(self.conv2d(h_cn8b, W_cn9) + b_cn9)

		# Tenth convolutional layers for the first signal
		with tf.name_scope('conv10a'):
			h_cn10a = tf.nn.relu(self.conv2d(h_cn9a, W_cn10) + b_cn10)

		# Tenth convolutional layers for the second signal
		with tf.name_scope('conv10b'):
			h_cn10b = tf.nn.relu(self.conv2d(h_cn9b, W_cn10) + b_cn10)

		# Fourth pooling layer for the first signal
		with tf.name_scope('pool4a'):
			h_pool4a = self.max_pool_2x2(h_cn10a)

		# Fourth pooling layer for the second signal
		with tf.name_scope('pool4b'):
			h_pool4b = self.max_pool_2x2(h_cn10b)

		# Eleventh convolutional layers for the first signal
		with tf.name_scope('conv11a'):
			h_cn11a = tf.nn.relu(self.conv2d(h_pool4a, W_cn11) + b_cn11)

		# Elevent convolutional layers for the second signal
		with tf.name_scope('conv11b'):
			h_cn11b = tf.nn.relu(self.conv2d(h_pool4b, W_cn11) + b_cn11)

		# Twelveth convolutional layers for the first signal
		with tf.name_scope('conv12a'):
			h_cn12a = tf.nn.relu(self.conv2d(h_cn11a, W_cn12) + b_cn12)

		# Twelveth convolutional layers for the second signal
		with tf.name_scope('conv12b'):
			h_cn12b = tf.nn.relu(self.conv2d(h_cn11b, W_cn12) + b_cn12)

		# Thirteen convolutional layers for the first signal
		with tf.name_scope('conv13a'):
			h_cn13a = tf.nn.relu(self.conv2d(h_cn12a, W_cn13) + b_cn13)

		# Thirteen convolutional layers for the second signal
		with tf.name_scope('conv13b'):
			h_cn13b = tf.nn.relu(self.conv2d(h_cn12b, W_cn13) + b_cn13)			 

		# Fourth pooling layer for the first signal
		with tf.name_scope('pool5a'):
			h_pool5a = self.max_pool_2x2(h_cn13a)

		# Second pooling layer for the second signal
		with tf.name_scope('pool5b'):
			h_pool5b = self.max_pool_2x2(h_cn13b)

		# Concat layer to go from convolutional layer to fully connected
		with tf.name_scope('concat1'):
			h_concat1 = tf.concat([h_pool5a, h_pool5b], axis=3)

		# First fully connected layer
		with tf.name_scope('fc1'):
			h_concat1_flat = tf.reshape(h_concat1, [-1, self.height_after_conv * self.width_after_conv *512*2]) 
			h_mat =	tf.matmul(h_concat1_flat, W_fc1)
			h_fc1 = tf.nn.relu(h_mat + b_fc1)

		# Second fully connected layer
		with tf.name_scope('fc2'):
			h_fc2 = tf.nn.relu(tf.matmul(h_fc1, W_fc2) + b_fc2)

		# Third fully connected layer
		with tf.name_scope('fc3'):
			self.Y_logt = tf.matmul(h_fc2, W_fc3) + b_fc3


class R50(Model):
	def __init_(self, 
		in_height,
		in_width,
		channels,
		label,
		pool_layers):
		Model.__init__(in_height,in_width,channels,label,pool_layers)

	def conv2ds2(self, x, W):
		# Defining the conv2d operation with a stride of 2
		return tf.nn.conv2d(x, W, strides=[1, 2, 2, 1], padding='SAME')

	def avg_pool_2x2(self,x):
		# Defining the average pool operation with a 7 x 7 kernel size and stride of 2
		return tf.nn.avg_pool(x, ksize=[1, 7, 7, 1],strides=[1, 2, 2, 1], padding='SAME')	

	def def_loss(self):
		""" Defines loss function """
		with tf.name_scope('loss'):

			_WEIGHT_DECAY = 0.01

			#cross entropy
			self.cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=self.Y, logits=self.Y_logt)
			self.loss = tf.reduce_mean(self.cross_entropy)
			regularizer = 0.0
			for weight in self.weight:
				regularizer = regularizer + tf.nn.l2_loss(self.weight[weight])
			self.loss = tf.reduce_mean(self.loss + _WEIGHT_DECAY * regularizer)

			# Alternative with regularizer only for the FC1
			# self.cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=self.Y, logits=self.Y_logt)
			# self.loss = tf.reduce_mean(self.cross_entropy)
			# regularizer = tf.nn.l2_loss(self.weight["W_fc1"])
			# self.loss = tf.reduce_mean(self.loss + _WEIGHT_DECAY * regularizer)

	def def_params(self):
		# Defining the parameters aka weights for the model
		# In this case the parameters are name with W for weight, b plus a number which means the number of the block
		# u plus a number which means the number of the unit of the block and cn plus a number which mean the convolutional
		# layer of the unit
		self.weight = {}

		""" Defines model parameters """
		with tf.name_scope('params'):

			# Zero convolutional layer
			with tf.name_scope('conv0'):			
				self.weight["W_cn0"] = self.weight_variable([7,7,1, 64])

			# Block 1 --> 3 Units, the first unit has a shortcut

			# Block 1, unit 1
			with tf.name_scope('block1_unit1'):
				self.weight["W_b1_u1_cn0"] = self.weight_variable([1,1,64,256])
				self.weight["W_b1_u1_cn1"] = self.weight_variable([1,1,64,64])
				self.weight["W_b1_u1_cn2"] = self.weight_variable([3,3,64,64])
				self.weight["W_b1_u1_cn3"] = self.weight_variable([1,1,64,256])

			# Block 1, unit 2
			with tf.name_scope('block1_unit2'):
				self.weight["W_b1_u2_cn1"] = self.weight_variable([1,1,256,64])
				self.weight["W_b1_u2_cn2"] = self.weight_variable([3,3,64,64])
				self.weight["W_b1_u2_cn3"] = self.weight_variable([1,1,64,256])

			# Block 1, unit 3
			with tf.name_scope('block1_unit3'):
				self.weight["W_b1_u3_cn1"] = self.weight_variable([1,1,256,64])
				self.weight["W_b1_u3_cn2"] = self.weight_variable([3,3,64,64])
				self.weight["W_b1_u3_cn3"] = self.weight_variable([1,1,64,256])


			# Block 2 --> 4 Units, the first unit has a shortcut

			# Block 2, unit 1
			with tf.name_scope('block2_unit1'):
				self.weight["W_b2_u1_cn0"] = self.weight_variable([1,1,256, 512])
				self.weight["W_b2_u1_cn1"] = self.weight_variable([1,1,256, 128])
				self.weight["W_b2_u1_cn2"] = self.weight_variable([3,3,128, 128])
				self.weight["W_b2_u1_cn3"] = self.weight_variable([1,1,128, 512])

			# Block 2, unit 2
			with tf.name_scope('block2_unit2'):
				self.weight["W_b2_u2_cn1"] = self.weight_variable([1,1,512, 128])
				self.weight["W_b2_u2_cn2"] = self.weight_variable([3,3,128, 128])
				self.weight["W_b2_u2_cn3"] = self.weight_variable([1,1,128, 512])

			# Block 2, unit 3
			with tf.name_scope('block2_unit3'):
				self.weight["W_b2_u3_cn1"] = self.weight_variable([1,1,512, 128])
				self.weight["W_b2_u3_cn2"] = self.weight_variable([3,3,128, 128])
				self.weight["W_b2_u3_cn3"] = self.weight_variable([1,1,128, 512])

			# Block 2, unit 4
			with tf.name_scope('block2_unit4'):
				self.weight["W_b2_u4_cn1"] = self.weight_variable([1,1,512, 128])
				self.weight["W_b2_u4_cn2"] = self.weight_variable([3,3,128, 128])
				self.weight["W_b2_u4_cn3"] = self.weight_variable([1,1,128, 512])


			# Block 3 --> 6 Units, the first unit has a shortcut

			# Block 3, unit 1
			with tf.name_scope('block3_unit1'):
				self.weight["W_b3_u1_cn0"] = self.weight_variable([1,1,512, 1024])
				self.weight["W_b3_u1_cn1"] = self.weight_variable([1,1,512, 256])
				self.weight["W_b3_u1_cn2"] = self.weight_variable([3,3,256, 256])
				self.weight["W_b3_u1_cn3"] = self.weight_variable([1,1,256, 1024])

			# Block 3, unit 2
			with tf.name_scope('block3_unit2'):
				self.weight["W_b3_u2_cn1"] = self.weight_variable([1,1,1024, 256])
				self.weight["W_b3_u2_cn2"] = self.weight_variable([3,3,256, 256])
				self.weight["W_b3_u2_cn3"] = self.weight_variable([1,1,256, 1024])

			# Block 3, unit 3
			with tf.name_scope('block3_unit3'):
				self.weight["W_b3_u3_cn1"] = self.weight_variable([1,1,1024, 256])
				self.weight["W_b3_u3_cn2"] = self.weight_variable([3,3,256, 256])
				self.weight["W_b3_u3_cn3"] = self.weight_variable([1,1,256, 1024])

			# Block 3, unit 4
			with tf.name_scope('block3_unit4'):
				self.weight["W_b3_u4_cn1"] = self.weight_variable([1,1,1024, 256])
				self.weight["W_b3_u4_cn2"] = self.weight_variable([3,3,256, 256])
				self.weight["W_b3_u4_cn3"] = self.weight_variable([1,1,256, 1024])

			# Block 3, unit 5
			with tf.name_scope('block3_unit5'):
				self.weight["W_b3_u5_cn1"] = self.weight_variable([1,1,1024, 256])
				self.weight["W_b3_u5_cn2"] = self.weight_variable([3,3,256, 256])
				self.weight["W_b3_u5_cn3"] = self.weight_variable([1,1,256, 1024])

			# Block 3, unit 6
			with tf.name_scope('block3_unit6'):
				self.weight["W_b3_u6_cn1"] = self.weight_variable([1,1,1024, 256])
				self.weight["W_b3_u6_cn2"] = self.weight_variable([3,3,256, 256])
				self.weight["W_b3_u6_cn3"] = self.weight_variable([1,1,256, 1024])


			# Block 4 --> 3 Units, the first unit has a shortcut

			# Block 4, unit 1
			with tf.name_scope('block4_unit1'):
				self.weight["W_b4_u1_cn0"] = self.weight_variable([1,1,1024, 2048])
				self.weight["W_b4_u1_cn1"] = self.weight_variable([1,1,1024, 512])
				self.weight["W_b4_u1_cn2"] = self.weight_variable([3,3,512, 512])
				self.weight["W_b4_u1_cn3"] = self.weight_variable([1,1,512, 2048])

			# Block 4, unit 2
			with tf.name_scope('block4_unit2'):
				self.weight["W_b4_u2_cn1"] = self.weight_variable([1,1,2048, 512])
				self.weight["W_b4_u2_cn2"] = self.weight_variable([3,3,512, 512])
				self.weight["W_b4_u2_cn3"] = self.weight_variable([1,1,512, 2048])

			# Block 4, unit 3
			with tf.name_scope('block4_unit3'):
				self.weight["W_b4_u3_cn1"] = self.weight_variable([1,1,2048, 512])
				self.weight["W_b4_u3_cn2"] = self.weight_variable([3,3,512, 512])
				self.weight["W_b4_u3_cn3"] = self.weight_variable([1,1,512, 2048])


			# Fully connected
			with tf.name_scope('fc1'):# 30 x 71
				self.weight["W_fc1"] = self.weight_variable([2 * 2048 * self.width_after_conv * self.height_after_conv, 2048])
				self.weight["W_fc2"] = self.weight_variable([2048, self.label])

	def def_model(self):
		""" Defines the model """
		with tf.name_scope('model'):

			with tf.name_scope('conv0a'):
				h_cn0a = self.conv2ds2(self.X1, self.weight["W_cn0"])
				h_cn0a = tf.layers.batch_normalization(inputs=h_cn0a , axis=3, momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True, scale=True, training=self.training, fused=True)
				h_cn0a = tf.nn.relu(h_cn0a)

			with tf.name_scope('pool0a'):
				h_pool1a = self.max_pool_2x2(h_cn0a)

			# Block 1, unit 1
			with tf.name_scope('block1_unit1a'):

				# Calculating the first shortcut
				shortcut_b1a = self.conv2d(h_pool1a, self.weight["W_b1_u1_cn0"])
				shortcut_b1a = tf.layers.batch_normalization(inputs=shortcut_b1a , axis=3, momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True, scale=True, training=self.training, fused=True)

				h_b1_u1_cn1a = self.conv2d(h_pool1a, self.weight["W_b1_u1_cn1"])
				h_b1_u1_cn1a = tf.layers.batch_normalization(inputs=h_b1_u1_cn1a , axis=3, momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True, scale=True, training=self.training, fused=True)
				h_b1_u1_cn1a = tf.nn.relu(h_b1_u1_cn1a)

				h_b1_u1_cn2a = self.conv2d(h_b1_u1_cn1a, self.weight["W_b1_u1_cn2"])
				h_b1_u1_cn2a = tf.layers.batch_normalization(inputs=h_b1_u1_cn2a , axis=3, momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True, scale=True, training=self.training, fused=True)
				h_b1_u1_cn2a = tf.nn.relu(h_b1_u1_cn2a)

				h_b1_u1_cn3a = self.conv2d(h_b1_u1_cn2a, self.weight["W_b1_u1_cn3"])
				h_b1_u1_cn3a = tf.layers.batch_normalization(inputs=h_b1_u1_cn3a , axis=3, momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True, scale=True, training=self.training, fused=True)
				h_b1_u1_cn3a = tf.add(h_b1_u1_cn3a, shortcut_b1a)
				h_b1_u1_cn3a = tf.nn.relu(h_b1_u1_cn3a)


			# Block 1, unit 2
			with tf.name_scope('block1_unit2a'):

				h_b1_u2_cn1a = self.conv2d(h_b1_u1_cn3a, self.weight["W_b1_u2_cn1"])
				h_b1_u2_cn1a = tf.layers.batch_normalization(inputs=h_b1_u2_cn1a , axis=3, momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True, scale=True, training=self.training, fused=True)
				h_b1_u2_cn1a = tf.nn.relu(h_b1_u2_cn1a)

				h_b1_u2_cn2a = self.conv2d(h_b1_u2_cn1a, self.weight["W_b1_u2_cn2"])
				h_b1_u2_cn2a = tf.layers.batch_normalization(inputs=h_b1_u2_cn2a , axis=3, momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True, scale=True, training=self.training, fused=True)
				h_b1_u2_cn2a = tf.nn.relu(h_b1_u2_cn2a)

				h_b1_u2_cn3a = self.conv2d(h_b1_u2_cn2a, self.weight["W_b1_u2_cn3"])
				h_b1_u2_cn3a = tf.layers.batch_normalization(inputs=h_b1_u2_cn3a , axis=3, momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True, scale=True, training=self.training, fused=True)
				h_b1_u2_cn3a = tf.add(h_b1_u2_cn3a, h_b1_u1_cn3a)
				h_b1_u2_cn3a = tf.nn.relu(h_b1_u2_cn3a)


			# Block 1, unit 3
			with tf.name_scope('block1_unit3a'):

				h_b1_u3_cn1a = self.conv2d(h_b1_u2_cn3a, self.weight["W_b1_u3_cn1"])
				h_b1_u3_cn1a = tf.layers.batch_normalization(inputs=h_b1_u3_cn1a , axis=3, momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True, scale=True, training=self.training, fused=True)
				h_b1_u3_cn1a = tf.nn.relu(h_b1_u3_cn1a)

				h_b1_u3_cn2a = self.conv2d(h_b1_u3_cn1a, self.weight["W_b1_u3_cn2"])
				h_b1_u3_cn2a = tf.layers.batch_normalization(inputs=h_b1_u3_cn2a , axis=3, momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True, scale=True, training=self.training, fused=True)
				h_b1_u3_cn2a = tf.nn.relu(h_b1_u3_cn2a)

				h_b1_u3_cn3a = self.conv2d(h_b1_u3_cn2a, self.weight["W_b1_u3_cn3"])
				h_b1_u3_cn3a = tf.layers.batch_normalization(inputs=h_b1_u3_cn3a , axis=3, momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True, scale=True, training=self.training, fused=True)
				h_b1_u3_cn3a = tf.add(h_b1_u3_cn3a, h_b1_u2_cn3a)
				h_b1_u3_cn3a = tf.nn.relu(h_b1_u3_cn3a)


			# Block 2, unit 1
			with tf.name_scope('block2_unit1a'):

				# Original way to go on a resnet50
				# Calculating the first shortcut
				# shortcut_b2a = self.conv2ds2(h_b1_u3_cn3a, self.weight["W_b2_u1_cn0"])
				# shortcut_b2a = tf.layers.batch_normalization(inputs=shortcut_b2a , axis=3, momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True, scale=True, training=self.training, fused=True)

				# h_b2_u1_cn1a = self.conv2ds2(h_b1_u3_cn3a, self.weight["W_b2_u1_cn1"])
				# h_b2_u1_cn1a = tf.layers.batch_normalization(inputs=h_b2_u1_cn1a , axis=3, momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True, scale=True, training=self.training, fused=True)
				# h_b2_u1_cn1a = tf.nn.relu(h_b2_u1_cn1a)

				# Modification in the resnet 50 due to excesive reduction of the input data through the blocks
				# The modification is to use the conv2d function insted of the conv2ds2

				# Calculating the first shortcut
				shortcut_b2a = self.conv2d(h_b1_u3_cn3a, self.weight["W_b2_u1_cn0"])
				shortcut_b2a = tf.layers.batch_normalization(inputs=shortcut_b2a , axis=3, momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True, scale=True, training=self.training, fused=True)

				h_b2_u1_cn1a = self.conv2d(h_b1_u3_cn3a, self.weight["W_b2_u1_cn1"])
				h_b2_u1_cn1a = tf.layers.batch_normalization(inputs=h_b2_u1_cn1a , axis=3, momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True, scale=True, training=self.training, fused=True)
				h_b2_u1_cn1a = tf.nn.relu(h_b2_u1_cn1a)

				h_b2_u1_cn2a = self.conv2d(h_b2_u1_cn1a, self.weight["W_b2_u1_cn2"])
				h_b2_u1_cn2a = tf.layers.batch_normalization(inputs=h_b2_u1_cn2a , axis=3, momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True, scale=True, training=self.training, fused=True)
				h_b2_u1_cn2a = tf.nn.relu(h_b2_u1_cn2a)

				h_b2_u1_cn3a = self.conv2d(h_b2_u1_cn2a, self.weight["W_b2_u1_cn3"])
				h_b2_u1_cn3a = tf.layers.batch_normalization(inputs=h_b2_u1_cn3a , axis=3, momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True, scale=True, training=self.training, fused=True)
				h_b2_u1_cn3a = tf.add(h_b2_u1_cn3a, shortcut_b2a)
				h_b2_u1_cn3a = tf.nn.relu(h_b2_u1_cn3a)

			
			# Block 2, unit 2
			with tf.name_scope('block2_unit2a'):

				h_b2_u2_cn1a = self.conv2d(h_b2_u1_cn3a, self.weight["W_b2_u2_cn1"])
				h_b2_u2_cn1a = tf.layers.batch_normalization(inputs=h_b2_u2_cn1a , axis=3, momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True, scale=True, training=self.training, fused=True)
				h_b2_u2_cn1a = tf.nn.relu(h_b2_u2_cn1a)

				h_b2_u2_cn2a = self.conv2d(h_b2_u2_cn1a, self.weight["W_b2_u2_cn2"])
				h_b2_u2_cn2a = tf.layers.batch_normalization(inputs=h_b2_u2_cn2a , axis=3, momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True, scale=True, training=self.training, fused=True)
				h_b2_u2_cn2a = tf.nn.relu(h_b2_u2_cn2a)

				h_b2_u2_cn3a = self.conv2d(h_b2_u2_cn2a, self.weight["W_b2_u2_cn3"])
				h_b2_u2_cn3a = tf.layers.batch_normalization(inputs=h_b2_u2_cn3a , axis=3, momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True, scale=True, training=self.training, fused=True)
				h_b2_u2_cn3a = tf.add(h_b2_u2_cn3a, h_b2_u1_cn3a)
				h_b2_u2_cn3a = tf.nn.relu(h_b2_u2_cn3a)


			# Block 2, unit 3
			with tf.name_scope('block2_unit3a'):

				h_b2_u3_cn1a = self.conv2d(h_b2_u2_cn3a, self.weight["W_b2_u3_cn1"])
				h_b2_u3_cn1a = tf.layers.batch_normalization(inputs=h_b2_u3_cn1a , axis=3, momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True, scale=True, training=self.training, fused=True)
				h_b2_u3_cn1a = tf.nn.relu(h_b2_u3_cn1a)

				h_b2_u3_cn2a = self.conv2d(h_b2_u3_cn1a, self.weight["W_b2_u3_cn2"])
				h_b2_u3_cn2a = tf.layers.batch_normalization(inputs=h_b2_u3_cn2a , axis=3, momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True, scale=True, training=self.training, fused=True)
				h_b2_u3_cn2a = tf.nn.relu(h_b2_u3_cn2a)

				h_b2_u3_cn3a = self.conv2d(h_b2_u3_cn2a, self.weight["W_b2_u3_cn3"])
				h_b2_u3_cn3a = tf.layers.batch_normalization(inputs=h_b2_u3_cn3a , axis=3, momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True, scale=True, training=self.training, fused=True)
				h_b2_u3_cn3a = tf.add(h_b2_u3_cn3a, h_b2_u2_cn3a)
				h_b2_u3_cn3a = tf.nn.relu(h_b2_u3_cn3a)


			# Block 2, unit 4
			with tf.name_scope('block2_unit4a'):

				h_b2_u4_cn1a = self.conv2d(h_b2_u3_cn3a, self.weight["W_b2_u4_cn1"])
				h_b2_u4_cn1a = tf.layers.batch_normalization(inputs=h_b2_u4_cn1a , axis=3, momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True, scale=True, training=self.training, fused=True)
				h_b2_u4_cn1a = tf.nn.relu(h_b2_u4_cn1a)

				h_b2_u4_cn2a = self.conv2d(h_b2_u4_cn1a, self.weight["W_b2_u4_cn2"])
				h_b2_u4_cn2a = tf.layers.batch_normalization(inputs=h_b2_u4_cn2a , axis=3, momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True, scale=True, training=self.training, fused=True)
				h_b2_u4_cn2a = tf.nn.relu(h_b2_u4_cn2a)

				h_b2_u4_cn3a = self.conv2d(h_b2_u4_cn2a, self.weight["W_b2_u4_cn3"])
				h_b2_u4_cn3a = tf.layers.batch_normalization(inputs=h_b2_u4_cn3a , axis=3, momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True, scale=True, training=self.training, fused=True)
				h_b2_u4_cn3a = tf.add(h_b2_u4_cn3a, h_b2_u3_cn3a)
				h_b2_u4_cn3a = tf.nn.relu(h_b2_u4_cn3a)


			# Block 3, unit 1
			with tf.name_scope('block3_unit1a'):

				# Original way to go on a resnet50
				# Calculating the first shortcut
				# shortcut_b3a = self.conv2ds2(h_b2_u4_cn3a, self.weight["W_b3_u1_cn0"])
				# shortcut_b3a = tf.layers.batch_normalization(inputs=shortcut_b3a , axis=3, momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True, scale=True, training=self.training, fused=True)

				# h_b3_u1_cn1a = self.conv2ds2(h_b2_u4_cn3a, self.weight["W_b3_u1_cn1"])
				# h_b3_u1_cn1a = tf.layers.batch_normalization(inputs=h_b3_u1_cn1a , axis=3, momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True, scale=True, training=self.training, fused=True)
				# h_b3_u1_cn1a = tf.nn.relu(h_b3_u1_cn1a)

				# Modification in the resnet 50 due to excesive reduction of the input data through the blocks
				# The modification is to use the conv2d function insted of the conv2ds2

				# Calculating the first shortcut
				shortcut_b3a = self.conv2d(h_b2_u4_cn3a, self.weight["W_b3_u1_cn0"])
				shortcut_b3a = tf.layers.batch_normalization(inputs=shortcut_b3a , axis=3, momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True, scale=True, training=self.training, fused=True)

				h_b3_u1_cn1a = self.conv2d(h_b2_u4_cn3a, self.weight["W_b3_u1_cn1"])
				h_b3_u1_cn1a = tf.layers.batch_normalization(inputs=h_b3_u1_cn1a , axis=3, momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True, scale=True, training=self.training, fused=True)
				h_b3_u1_cn1a = tf.nn.relu(h_b3_u1_cn1a)

				h_b3_u1_cn2a = self.conv2d(h_b3_u1_cn1a, self.weight["W_b3_u1_cn2"])
				h_b3_u1_cn2a = tf.layers.batch_normalization(inputs=h_b3_u1_cn2a , axis=3, momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True, scale=True, training=self.training, fused=True)
				h_b3_u1_cn2a = tf.nn.relu(h_b3_u1_cn2a)

				h_b3_u1_cn3a = self.conv2d(h_b3_u1_cn2a, self.weight["W_b3_u1_cn3"])
				h_b3_u1_cn3a = tf.layers.batch_normalization(inputs=h_b3_u1_cn3a , axis=3, momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True, scale=True, training=self.training, fused=True)
				h_b3_u1_cn3a = tf.add(h_b3_u1_cn3a, shortcut_b3a)
				h_b3_u1_cn3a = tf.nn.relu(h_b3_u1_cn3a)

			
			# Block 3, unit 2
			with tf.name_scope('block3_unit2a'):

				h_b3_u2_cn1a = self.conv2d(h_b3_u1_cn3a, self.weight["W_b3_u2_cn1"])
				h_b3_u2_cn1a = tf.layers.batch_normalization(inputs=h_b3_u2_cn1a , axis=3, momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True, scale=True, training=self.training, fused=True)
				h_b3_u2_cn1a = tf.nn.relu(h_b3_u2_cn1a)

				h_b3_u2_cn2a = self.conv2d(h_b3_u2_cn1a, self.weight["W_b3_u2_cn2"])
				h_b3_u2_cn2a = tf.layers.batch_normalization(inputs=h_b3_u2_cn2a , axis=3, momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True, scale=True, training=self.training, fused=True)
				h_b3_u2_cn2a = tf.nn.relu(h_b3_u2_cn2a)

				h_b3_u2_cn3a = self.conv2d(h_b3_u2_cn2a, self.weight["W_b3_u2_cn3"])
				h_b3_u2_cn3a = tf.layers.batch_normalization(inputs=h_b3_u2_cn3a , axis=3, momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True, scale=True, training=self.training, fused=True)
				h_b3_u2_cn3a = tf.add(h_b3_u2_cn3a, h_b3_u1_cn3a)
				h_b3_u2_cn3a = tf.nn.relu(h_b3_u2_cn3a)


			# Block 3, unit 3
			with tf.name_scope('block3_unit3a'):

				h_b3_u3_cn1a = self.conv2d(h_b3_u2_cn3a, self.weight["W_b3_u3_cn1"])
				h_b3_u3_cn1a = tf.layers.batch_normalization(inputs=h_b3_u3_cn1a , axis=3, momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True, scale=True, training=self.training, fused=True)
				h_b3_u3_cn1a = tf.nn.relu(h_b3_u3_cn1a)

				h_b3_u3_cn2a = self.conv2d(h_b3_u3_cn1a, self.weight["W_b3_u3_cn2"])
				h_b3_u3_cn2a = tf.layers.batch_normalization(inputs=h_b3_u3_cn2a , axis=3, momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True, scale=True, training=self.training, fused=True)
				h_b3_u3_cn2a = tf.nn.relu(h_b3_u3_cn2a)

				h_b3_u3_cn3a = self.conv2d(h_b3_u3_cn2a, self.weight["W_b3_u3_cn3"])
				h_b3_u3_cn3a = tf.layers.batch_normalization(inputs=h_b3_u3_cn3a , axis=3, momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True, scale=True, training=self.training, fused=True)
				h_b3_u3_cn3a = tf.add(h_b3_u3_cn3a, h_b3_u2_cn3a)
				h_b3_u3_cn3a = tf.nn.relu(h_b3_u3_cn3a)


			# Block 3, unit 4
			with tf.name_scope('block3_unit4a'):

				h_b3_u4_cn1a = self.conv2d(h_b3_u3_cn3a, self.weight["W_b3_u4_cn1"])
				h_b3_u4_cn1a = tf.layers.batch_normalization(inputs=h_b3_u4_cn1a , axis=3, momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True, scale=True, training=self.training, fused=True)
				h_b3_u4_cn1a = tf.nn.relu(h_b3_u4_cn1a)

				h_b3_u4_cn2a = self.conv2d(h_b3_u4_cn1a, self.weight["W_b3_u4_cn2"])
				h_b3_u4_cn2a = tf.layers.batch_normalization(inputs=h_b3_u4_cn2a , axis=3, momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True, scale=True, training=self.training, fused=True)
				h_b3_u4_cn2a = tf.nn.relu(h_b3_u4_cn2a)

				h_b3_u4_cn3a = self.conv2d(h_b3_u4_cn2a, self.weight["W_b3_u4_cn3"])
				h_b3_u4_cn3a = tf.layers.batch_normalization(inputs=h_b3_u4_cn3a , axis=3, momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True, scale=True, training=self.training, fused=True)
				h_b3_u4_cn3a = tf.add(h_b3_u4_cn3a, h_b3_u3_cn3a)
				h_b3_u4_cn3a = tf.nn.relu(h_b3_u4_cn3a)


			# Block 3, unit 5
			with tf.name_scope('block3_unit5a'):

				h_b3_u5_cn1a = self.conv2d(h_b3_u4_cn3a, self.weight["W_b3_u5_cn1"])
				h_b3_u5_cn1a = tf.layers.batch_normalization(inputs=h_b3_u5_cn1a , axis=3, momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True, scale=True, training=self.training, fused=True)
				h_b3_u5_cn1a = tf.nn.relu(h_b3_u5_cn1a)

				h_b3_u5_cn2a = self.conv2d(h_b3_u5_cn1a, self.weight["W_b3_u5_cn2"])
				h_b3_u5_cn2a = tf.layers.batch_normalization(inputs=h_b3_u5_cn2a , axis=3, momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True, scale=True, training=self.training, fused=True)
				h_b3_u5_cn2a = tf.nn.relu(h_b3_u5_cn2a)

				h_b3_u5_cn3a = self.conv2d(h_b3_u5_cn2a, self.weight["W_b3_u5_cn3"])
				h_b3_u5_cn3a = tf.layers.batch_normalization(inputs=h_b3_u5_cn3a , axis=3, momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True, scale=True, training=self.training, fused=True)
				h_b3_u5_cn3a = tf.add(h_b3_u5_cn3a, h_b3_u4_cn3a)
				h_b3_u5_cn3a = tf.nn.relu(h_b3_u5_cn3a)


			# Block 3, unit 6
			with tf.name_scope('block3_unit6a'):

				h_b3_u6_cn1a = self.conv2d(h_b3_u5_cn3a, self.weight["W_b3_u6_cn1"])
				h_b3_u6_cn1a = tf.layers.batch_normalization(inputs=h_b3_u6_cn1a , axis=3, momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True, scale=True, training=self.training, fused=True)
				h_b3_u6_cn1a = tf.nn.relu(h_b3_u6_cn1a)

				h_b3_u6_cn2a = self.conv2d(h_b3_u6_cn1a, self.weight["W_b3_u6_cn2"])
				h_b3_u6_cn2a = tf.layers.batch_normalization(inputs=h_b3_u6_cn2a , axis=3, momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True, scale=True, training=self.training, fused=True)
				h_b3_u6_cn2a = tf.nn.relu(h_b3_u6_cn2a)

				h_b3_u6_cn3a = self.conv2d(h_b3_u6_cn2a, self.weight["W_b3_u6_cn3"])
				h_b3_u6_cn3a = tf.layers.batch_normalization(inputs=h_b3_u6_cn3a , axis=3, momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True, scale=True, training=self.training, fused=True)
				h_b3_u6_cn3a = tf.add(h_b3_u6_cn3a, h_b3_u5_cn3a)
				h_b3_u6_cn3a = tf.nn.relu(h_b3_u6_cn3a)


			# Block 4, unit 1
			with tf.name_scope('block4_unit1a'):

				# Original way to go on a resnet50
				# Calculating the first shortcut
				# shortcut_b4a = self.conv2ds2(h_b3_u6_cn3a, self.weight["W_b4_u1_cn0"])
				# shortcut_b4a = tf.layers.batch_normalization(inputs=shortcut_b4a , axis=3, momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True, scale=True, training=self.training, fused=True)

				# h_b4_u1_cn1a = self.conv2ds2(h_b3_u6_cn3a, self.weight["W_b4_u1_cn1"])
				# h_b4_u1_cn1a = tf.layers.batch_normalization(inputs=h_b4_u1_cn1a , axis=3, momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True, scale=True, training=self.training, fused=True)
				# h_b4_u1_cn1a = tf.nn.relu(h_b4_u1_cn1a)

				# Modification in the resnet 50 due to excesive reduction of the input data through the blocks
				# The modification is to use the conv2d function insted of the conv2ds2

				# Calculating the first shortcut
				shortcut_b4a = self.conv2d(h_b3_u6_cn3a, self.weight["W_b4_u1_cn0"])
				shortcut_b4a = tf.layers.batch_normalization(inputs=shortcut_b4a , axis=3, momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True, scale=True, training=self.training, fused=True)

				h_b4_u1_cn1a = self.conv2d(h_b3_u6_cn3a, self.weight["W_b4_u1_cn1"])
				h_b4_u1_cn1a = tf.layers.batch_normalization(inputs=h_b4_u1_cn1a , axis=3, momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True, scale=True, training=self.training, fused=True)
				h_b4_u1_cn1a = tf.nn.relu(h_b4_u1_cn1a)

				h_b4_u1_cn2a = self.conv2d(h_b4_u1_cn1a, self.weight["W_b4_u1_cn2"])
				h_b4_u1_cn2a = tf.layers.batch_normalization(inputs=h_b4_u1_cn2a , axis=3, momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True, scale=True, training=self.training, fused=True)
				h_b4_u1_cn2a = tf.nn.relu(h_b4_u1_cn2a)

				h_b4_u1_cn3a = self.conv2d(h_b4_u1_cn2a, self.weight["W_b4_u1_cn3"])
				h_b4_u1_cn3a = tf.layers.batch_normalization(inputs=h_b4_u1_cn3a , axis=3, momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True, scale=True, training=self.training, fused=True)
				h_b4_u1_cn3a = tf.add(h_b4_u1_cn3a, shortcut_b4a)
				h_b4_u1_cn3a = tf.nn.relu(h_b4_u1_cn3a)


			# Block 4, unit 2
			with tf.name_scope('block4_unit2a'):

				h_b4_u2_cn1a = self.conv2d(h_b4_u1_cn3a, self.weight["W_b4_u2_cn1"])
				h_b4_u2_cn1a = tf.layers.batch_normalization(inputs=h_b4_u2_cn1a , axis=3, momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True, scale=True, training=self.training, fused=True)
				h_b4_u2_cn1a = tf.nn.relu(h_b4_u2_cn1a)

				h_b4_u2_cn2a = self.conv2d(h_b4_u2_cn1a, self.weight["W_b4_u2_cn2"])
				h_b4_u2_cn2a = tf.layers.batch_normalization(inputs=h_b4_u2_cn2a , axis=3, momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True, scale=True, training=self.training, fused=True)
				h_b4_u2_cn2a = tf.nn.relu(h_b4_u2_cn2a)

				h_b4_u2_cn3a = self.conv2d(h_b4_u2_cn2a, self.weight["W_b4_u2_cn3"])
				h_b4_u2_cn3a = tf.layers.batch_normalization(inputs=h_b4_u2_cn3a , axis=3, momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True, scale=True, training=self.training, fused=True)
				h_b4_u2_cn3a = tf.add(h_b4_u2_cn3a, h_b4_u1_cn3a)
				h_b4_u2_cn3a = tf.nn.relu(h_b4_u2_cn3a)


			# Block 4, unit 3
			with tf.name_scope('block4_unit3a'):

				h_b4_u3_cn1a = self.conv2d(h_b4_u2_cn3a, self.weight["W_b4_u3_cn1"])
				h_b4_u3_cn1a = tf.layers.batch_normalization(inputs=h_b4_u3_cn1a , axis=3, momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True, scale=True, training=self.training, fused=True)
				h_b4_u3_cn1a = tf.nn.relu(h_b4_u3_cn1a)

				h_b4_u3_cn2a = self.conv2d(h_b4_u3_cn1a, self.weight["W_b4_u3_cn2"])
				h_b4_u3_cn2a = tf.layers.batch_normalization(inputs=h_b4_u3_cn2a , axis=3, momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True, scale=True, training=self.training, fused=True)
				h_b4_u3_cn2a = tf.nn.relu(h_b4_u3_cn2a)

				h_b4_u3_cn3a = self.conv2d(h_b4_u3_cn2a, self.weight["W_b4_u3_cn3"])
				h_b4_u3_cn3a = tf.layers.batch_normalization(inputs=h_b4_u3_cn3a , axis=3, momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True, scale=True, training=self.training, fused=True)
				h_b4_u3_cn3a = tf.add(h_b4_u3_cn3a, h_b4_u2_cn3a)
				h_b4_u3_cn3a = tf.nn.relu(h_b4_u3_cn3a)

			with tf.name_scope('pool1a'):
				h_pool2a = self.avg_pool_2x2(h_b4_u3_cn3a)
				


			with tf.name_scope('conv0b'):
				h_cn0b = self.conv2ds2(self.X2, self.weight["W_cn0"])
				h_cn0b = tf.layers.batch_normalization(inputs=h_cn0b , axis=3, momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True, scale=True, training=self.training, fused=True)
				h_cn0b = tf.nn.relu(h_cn0b)

			with tf.name_scope('pool0b'):
				h_pool1b = self.max_pool_2x2(h_cn0b)

			# Block 1, unit 1
			with tf.name_scope('block1_unit1b'):

				# Calculating the first shortcut
				shortcut_b1b = self.conv2d(h_pool1b, self.weight["W_b1_u1_cn0"])
				shortcut_b1b = tf.layers.batch_normalization(inputs=shortcut_b1b , axis=3, momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True, scale=True, training=self.training, fused=True)

				h_b1_u1_cn1b = self.conv2d(h_pool1b, self.weight["W_b1_u1_cn1"])
				h_b1_u1_cn1b = tf.layers.batch_normalization(inputs=h_b1_u1_cn1b , axis=3, momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True, scale=True, training=self.training, fused=True)
				h_b1_u1_cn1b = tf.nn.relu(h_b1_u1_cn1b)

				h_b1_u1_cn2b = self.conv2d(h_b1_u1_cn1b, self.weight["W_b1_u1_cn2"])
				h_b1_u1_cn2b = tf.layers.batch_normalization(inputs=h_b1_u1_cn2b , axis=3, momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True, scale=True, training=self.training, fused=True)
				h_b1_u1_cn2b = tf.nn.relu(h_b1_u1_cn2b)

				h_b1_u1_cn3b = self.conv2d(h_b1_u1_cn2b, self.weight["W_b1_u1_cn3"])
				h_b1_u1_cn3b = tf.layers.batch_normalization(inputs=h_b1_u1_cn3b , axis=3, momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True, scale=True, training=self.training, fused=True)
				h_b1_u1_cn3b = tf.add(h_b1_u1_cn3b, shortcut_b1b)
				h_b1_u1_cn3b = tf.nn.relu(h_b1_u1_cn3b)


			# Block 1, unit 2
			with tf.name_scope('block1_unit2b'):

				h_b1_u2_cn1b = self.conv2d(h_b1_u1_cn3b, self.weight["W_b1_u2_cn1"])
				h_b1_u2_cn1b = tf.layers.batch_normalization(inputs=h_b1_u2_cn1b , axis=3, momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True, scale=True, training=self.training, fused=True)
				h_b1_u2_cn1b = tf.nn.relu(h_b1_u2_cn1b)

				h_b1_u2_cn2b = self.conv2d(h_b1_u2_cn1b, self.weight["W_b1_u2_cn2"])
				h_b1_u2_cn2b = tf.layers.batch_normalization(inputs=h_b1_u2_cn2b , axis=3, momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True, scale=True, training=self.training, fused=True)
				h_b1_u2_cn2b = tf.nn.relu(h_b1_u2_cn2b)

				h_b1_u2_cn3b = self.conv2d(h_b1_u2_cn2b, self.weight["W_b1_u2_cn3"])
				h_b1_u2_cn3b = tf.layers.batch_normalization(inputs=h_b1_u2_cn3b , axis=3, momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True, scale=True, training=self.training, fused=True)
				h_b1_u2_cn3b = tf.add(h_b1_u2_cn3b, h_b1_u1_cn3b)
				h_b1_u2_cn3b = tf.nn.relu(h_b1_u2_cn3b)


			# Block 1, unit 3
			with tf.name_scope('block1_unit3b'):

				h_b1_u3_cn1b = self.conv2d(h_b1_u2_cn3b, self.weight["W_b1_u3_cn1"])
				h_b1_u3_cn1b = tf.layers.batch_normalization(inputs=h_b1_u3_cn1b , axis=3, momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True, scale=True, training=self.training, fused=True)
				h_b1_u3_cn1b = tf.nn.relu(h_b1_u3_cn1b)

				h_b1_u3_cn2b = self.conv2d(h_b1_u3_cn1b, self.weight["W_b1_u3_cn2"])
				h_b1_u3_cn2b = tf.layers.batch_normalization(inputs=h_b1_u3_cn2b , axis=3, momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True, scale=True, training=self.training, fused=True)
				h_b1_u3_cn2b = tf.nn.relu(h_b1_u3_cn2b)

				h_b1_u3_cn3b = self.conv2d(h_b1_u3_cn2b, self.weight["W_b1_u3_cn3"])
				h_b1_u3_cn3b = tf.layers.batch_normalization(inputs=h_b1_u3_cn3b , axis=3, momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True, scale=True, training=self.training, fused=True)
				h_b1_u3_cn3b = tf.add(h_b1_u3_cn3b, h_b1_u2_cn3b)
				h_b1_u3_cn3b = tf.nn.relu(h_b1_u3_cn3b)


			# Block 2, unit 1
			with tf.name_scope('block2_unit1b'):
				# Original way to go on a resnet50
				# Calculating the first shortcut
				# shortcut_b2b = self.conv2ds2(h_b1_u3_cn3b, self.weight["W_b2_u1_cn0"])
				# shortcut_b2b = tf.layers.batch_normalization(inputs=shortcut_b2b , axis=3, momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True, scale=True, training=self.training, fused=True)

				# h_b2_u1_cn1b = self.conv2ds2(h_b1_u3_cn3b, self.weight["W_b2_u1_cn1"])
				# h_b2_u1_cn1b = tf.layers.batch_normalization(inputs=h_b2_u1_cn1b , axis=3, momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True, scale=True, training=self.training, fused=True)
				# h_b2_u1_cn1b = tf.nn.relu(h_b2_u1_cn1b)


				# Modification in the resnet 50 due to excesive reduction of the input data through the blocks
				# The modification is to use the conv2d function insted of the conv2ds2

				# Calculating the first shortcut
				shortcut_b2b = self.conv2d(h_b1_u3_cn3b, self.weight["W_b2_u1_cn0"])
				shortcut_b2b = tf.layers.batch_normalization(inputs=shortcut_b2b , axis=3, momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True, scale=True, training=self.training, fused=True)

				h_b2_u1_cn1b = self.conv2d(h_b1_u3_cn3b, self.weight["W_b2_u1_cn1"])
				h_b2_u1_cn1b = tf.layers.batch_normalization(inputs=h_b2_u1_cn1b , axis=3, momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True, scale=True, training=self.training, fused=True)
				h_b2_u1_cn1b = tf.nn.relu(h_b2_u1_cn1b)

				h_b2_u1_cn2b = self.conv2d(h_b2_u1_cn1b, self.weight["W_b2_u1_cn2"])
				h_b2_u1_cn2b = tf.layers.batch_normalization(inputs=h_b2_u1_cn2b , axis=3, momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True, scale=True, training=self.training, fused=True)
				h_b2_u1_cn2b = tf.nn.relu(h_b2_u1_cn2b)

				h_b2_u1_cn3b = self.conv2d(h_b2_u1_cn2b, self.weight["W_b2_u1_cn3"])
				h_b2_u1_cn3b = tf.layers.batch_normalization(inputs=h_b2_u1_cn3b , axis=3, momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True, scale=True, training=self.training, fused=True)
				h_b2_u1_cn3b = tf.add(h_b2_u1_cn3b, shortcut_b2b)
				h_b2_u1_cn3b = tf.nn.relu(h_b2_u1_cn3b)

			
			# Block 2, unit 2
			with tf.name_scope('block2_unit2b'):

				h_b2_u2_cn1b = self.conv2d(h_b2_u1_cn3b, self.weight["W_b2_u2_cn1"])
				h_b2_u2_cn1b = tf.layers.batch_normalization(inputs=h_b2_u2_cn1b , axis=3, momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True, scale=True, training=self.training, fused=True)
				h_b2_u2_cn1b = tf.nn.relu(h_b2_u2_cn1b)

				h_b2_u2_cn2b = self.conv2d(h_b2_u2_cn1b, self.weight["W_b2_u2_cn2"])
				h_b2_u2_cn2b = tf.layers.batch_normalization(inputs=h_b2_u2_cn2b , axis=3, momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True, scale=True, training=self.training, fused=True)
				h_b2_u2_cn2b = tf.nn.relu(h_b2_u2_cn2b)

				h_b2_u2_cn3b = self.conv2d(h_b2_u2_cn2b, self.weight["W_b2_u2_cn3"])
				h_b2_u2_cn3b = tf.layers.batch_normalization(inputs=h_b2_u2_cn3b , axis=3, momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True, scale=True, training=self.training, fused=True)
				h_b2_u2_cn3b = tf.add(h_b2_u2_cn3b, h_b2_u1_cn3b)
				h_b2_u2_cn3b = tf.nn.relu(h_b2_u2_cn3b)


			# Block 2, unit 3
			with tf.name_scope('block2_unit3b'):

				h_b2_u3_cn1b = self.conv2d(h_b2_u2_cn3b, self.weight["W_b2_u3_cn1"])
				h_b2_u3_cn1b = tf.layers.batch_normalization(inputs=h_b2_u3_cn1b , axis=3, momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True, scale=True, training=self.training, fused=True)
				h_b2_u3_cn1b = tf.nn.relu(h_b2_u3_cn1b)

				h_b2_u3_cn2b = self.conv2d(h_b2_u3_cn1b, self.weight["W_b2_u3_cn2"])
				h_b2_u3_cn2b = tf.layers.batch_normalization(inputs=h_b2_u3_cn2b , axis=3, momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True, scale=True, training=self.training, fused=True)
				h_b2_u3_cn2b = tf.nn.relu(h_b2_u3_cn2b)

				h_b2_u3_cn3b = self.conv2d(h_b2_u3_cn2b, self.weight["W_b2_u3_cn3"])
				h_b2_u3_cn3b = tf.layers.batch_normalization(inputs=h_b2_u3_cn3b , axis=3, momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True, scale=True, training=self.training, fused=True)
				h_b2_u3_cn3b = tf.add(h_b2_u3_cn3b, h_b2_u2_cn3b)
				h_b2_u3_cn3b = tf.nn.relu(h_b2_u3_cn3b)


			# Block 2, unit 4
			with tf.name_scope('block2_unit4b'):

				h_b2_u4_cn1b = self.conv2d(h_b2_u3_cn3b, self.weight["W_b2_u4_cn1"])
				h_b2_u4_cn1b = tf.layers.batch_normalization(inputs=h_b2_u4_cn1b , axis=3, momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True, scale=True, training=self.training, fused=True)
				h_b2_u4_cn1b = tf.nn.relu(h_b2_u4_cn1b)

				h_b2_u4_cn2b = self.conv2d(h_b2_u4_cn1b, self.weight["W_b2_u4_cn2"])
				h_b2_u4_cn2b = tf.layers.batch_normalization(inputs=h_b2_u4_cn2b , axis=3, momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True, scale=True, training=self.training, fused=True)
				h_b2_u4_cn2b = tf.nn.relu(h_b2_u4_cn2b)

				h_b2_u4_cn3b = self.conv2d(h_b2_u4_cn2b, self.weight["W_b2_u4_cn3"])
				h_b2_u4_cn3b = tf.layers.batch_normalization(inputs=h_b2_u4_cn3b , axis=3, momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True, scale=True, training=self.training, fused=True)
				h_b2_u4_cn3b = tf.add(h_b2_u4_cn3b, h_b2_u3_cn3b)
				h_b2_u4_cn3b = tf.nn.relu(h_b2_u4_cn3b)


			# Block 3, unit 1
			with tf.name_scope('block3_unit1b'):

				# Original way to go on a resnet50
				# Calculating the first shortcut
				# shortcut_b3b = self.conv2ds2(h_b2_u4_cn3b, self.weight["W_b3_u1_cn0"])
				# shortcut_b3b = tf.layers.batch_normalization(inputs=shortcut_b3b , axis=3, momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True, scale=True, training=self.training, fused=True)

				# h_b3_u1_cn1b = self.conv2ds2(h_b2_u4_cn3b, self.weight["W_b3_u1_cn1"])
				# h_b3_u1_cn1b = tf.layers.batch_normalization(inputs=h_b3_u1_cn1b , axis=3, momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True, scale=True, training=self.training, fused=True)
				# h_b3_u1_cn1b = tf.nn.relu(h_b3_u1_cn1b)

				# Modification in the resnet 50 due to excesive reduction of the input data through the blocks
				# The modification is to use the conv2d function insted of the conv2ds2

				# Calculating the first shortcut
				shortcut_b3b = self.conv2d(h_b2_u4_cn3b, self.weight["W_b3_u1_cn0"])
				shortcut_b3b = tf.layers.batch_normalization(inputs=shortcut_b3b , axis=3, momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True, scale=True, training=self.training, fused=True)

				h_b3_u1_cn1b = self.conv2d(h_b2_u4_cn3b, self.weight["W_b3_u1_cn1"])
				h_b3_u1_cn1b = tf.layers.batch_normalization(inputs=h_b3_u1_cn1b , axis=3, momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True, scale=True, training=self.training, fused=True)
				h_b3_u1_cn1b = tf.nn.relu(h_b3_u1_cn1b)

				h_b3_u1_cn2b = self.conv2d(h_b3_u1_cn1b, self.weight["W_b3_u1_cn2"])
				h_b3_u1_cn2b = tf.layers.batch_normalization(inputs=h_b3_u1_cn2b , axis=3, momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True, scale=True, training=self.training, fused=True)
				h_b3_u1_cn2b = tf.nn.relu(h_b3_u1_cn2b)

				h_b3_u1_cn3b = self.conv2d(h_b3_u1_cn2b, self.weight["W_b3_u1_cn3"])
				h_b3_u1_cn3b = tf.layers.batch_normalization(inputs=h_b3_u1_cn3b , axis=3, momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True, scale=True, training=self.training, fused=True)
				h_b3_u1_cn3b = tf.add(h_b3_u1_cn3b, shortcut_b3b)
				h_b3_u1_cn3b = tf.nn.relu(h_b3_u1_cn3b)

			
			# Block 3, unit 2
			with tf.name_scope('block3_unit2b'):

				h_b3_u2_cn1b = self.conv2d(h_b3_u1_cn3b, self.weight["W_b3_u2_cn1"])
				h_b3_u2_cn1b = tf.layers.batch_normalization(inputs=h_b3_u2_cn1b , axis=3, momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True, scale=True, training=self.training, fused=True)
				h_b3_u2_cn1b = tf.nn.relu(h_b3_u2_cn1b)

				h_b3_u2_cn2b = self.conv2d(h_b3_u2_cn1b, self.weight["W_b3_u2_cn2"])
				h_b3_u2_cn2b = tf.layers.batch_normalization(inputs=h_b3_u2_cn2b , axis=3, momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True, scale=True, training=self.training, fused=True)
				h_b3_u2_cn2b = tf.nn.relu(h_b3_u2_cn2b)

				h_b3_u2_cn3b = self.conv2d(h_b3_u2_cn2b, self.weight["W_b3_u2_cn3"])
				h_b3_u2_cn3b = tf.layers.batch_normalization(inputs=h_b3_u2_cn3b , axis=3, momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True, scale=True, training=self.training, fused=True)
				h_b3_u2_cn3b = tf.add(h_b3_u2_cn3b, h_b3_u1_cn3b)
				h_b3_u2_cn3b = tf.nn.relu(h_b3_u2_cn3b)


			# Block 3, unit 3
			with tf.name_scope('block3_unit3b'):

				h_b3_u3_cn1b = self.conv2d(h_b3_u2_cn3b, self.weight["W_b3_u3_cn1"])
				h_b3_u3_cn1b = tf.layers.batch_normalization(inputs=h_b3_u3_cn1b , axis=3, momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True, scale=True, training=self.training, fused=True)
				h_b3_u3_cn1b = tf.nn.relu(h_b3_u3_cn1b)

				h_b3_u3_cn2b = self.conv2d(h_b3_u3_cn1b, self.weight["W_b3_u3_cn2"])
				h_b3_u3_cn2b = tf.layers.batch_normalization(inputs=h_b3_u3_cn2b , axis=3, momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True, scale=True, training=self.training, fused=True)
				h_b3_u3_cn2b = tf.nn.relu(h_b3_u3_cn2b)

				h_b3_u3_cn3b = self.conv2d(h_b3_u3_cn2b, self.weight["W_b3_u3_cn3"])
				h_b3_u3_cn3b = tf.layers.batch_normalization(inputs=h_b3_u3_cn3b , axis=3, momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True, scale=True, training=self.training, fused=True)
				h_b3_u3_cn3b = tf.add(h_b3_u3_cn3b, h_b3_u2_cn3b)
				h_b3_u3_cn3b = tf.nn.relu(h_b3_u3_cn3b)


			# Block 3, unit 4
			with tf.name_scope('block3_unit4b'):

				h_b3_u4_cn1b = self.conv2d(h_b3_u3_cn3b, self.weight["W_b3_u4_cn1"])
				h_b3_u4_cn1b = tf.layers.batch_normalization(inputs=h_b3_u4_cn1b , axis=3, momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True, scale=True, training=self.training, fused=True)
				h_b3_u4_cn1b = tf.nn.relu(h_b3_u4_cn1b)

				h_b3_u4_cn2b = self.conv2d(h_b3_u4_cn1b, self.weight["W_b3_u4_cn2"])
				h_b3_u4_cn2b = tf.layers.batch_normalization(inputs=h_b3_u4_cn2b , axis=3, momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True, scale=True, training=self.training, fused=True)
				h_b3_u4_cn2b = tf.nn.relu(h_b3_u4_cn2b)

				h_b3_u4_cn3b = self.conv2d(h_b3_u4_cn2b, self.weight["W_b3_u4_cn3"])
				h_b3_u4_cn3b = tf.layers.batch_normalization(inputs=h_b3_u4_cn3b , axis=3, momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True, scale=True, training=self.training, fused=True)
				h_b3_u4_cn3b = tf.add(h_b3_u4_cn3b, h_b3_u3_cn3b)
				h_b3_u4_cn3b = tf.nn.relu(h_b3_u4_cn3b)


			# Block 3, unit 5
			with tf.name_scope('block3_unit5b'):

				h_b3_u5_cn1b = self.conv2d(h_b3_u4_cn3b, self.weight["W_b3_u5_cn1"])
				h_b3_u5_cn1b = tf.layers.batch_normalization(inputs=h_b3_u5_cn1b , axis=3, momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True, scale=True, training=self.training, fused=True)
				h_b3_u5_cn1b = tf.nn.relu(h_b3_u5_cn1b)

				h_b3_u5_cn2b = self.conv2d(h_b3_u5_cn1b, self.weight["W_b3_u5_cn2"])
				h_b3_u5_cn2b = tf.layers.batch_normalization(inputs=h_b3_u5_cn2b , axis=3, momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True, scale=True, training=self.training, fused=True)
				h_b3_u5_cn2b = tf.nn.relu(h_b3_u5_cn2b)

				h_b3_u5_cn3b = self.conv2d(h_b3_u5_cn2b, self.weight["W_b3_u5_cn3"])
				h_b3_u5_cn3b = tf.layers.batch_normalization(inputs=h_b3_u5_cn3b , axis=3, momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True, scale=True, training=self.training, fused=True)
				h_b3_u5_cn3b = tf.add(h_b3_u5_cn3b, h_b3_u4_cn3b)
				h_b3_u5_cn3b = tf.nn.relu(h_b3_u5_cn3b)


			# Block 3, unit 6
			with tf.name_scope('block3_unit6b'):

				h_b3_u6_cn1b = self.conv2d(h_b3_u5_cn3b, self.weight["W_b3_u6_cn1"])
				h_b3_u6_cn1b = tf.layers.batch_normalization(inputs=h_b3_u6_cn1b , axis=3, momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True, scale=True, training=self.training, fused=True)
				h_b3_u6_cn1b = tf.nn.relu(h_b3_u6_cn1b)

				h_b3_u6_cn2b = self.conv2d(h_b3_u6_cn1b, self.weight["W_b3_u6_cn2"])
				h_b3_u6_cn2b = tf.layers.batch_normalization(inputs=h_b3_u6_cn2b , axis=3, momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True, scale=True, training=self.training, fused=True)
				h_b3_u6_cn2b = tf.nn.relu(h_b3_u6_cn2b)

				h_b3_u6_cn3b = self.conv2d(h_b3_u6_cn2b, self.weight["W_b3_u6_cn3"])
				h_b3_u6_cn3b = tf.layers.batch_normalization(inputs=h_b3_u6_cn3b , axis=3, momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True, scale=True, training=self.training, fused=True)
				h_b3_u6_cn3b = tf.add(h_b3_u6_cn3b, h_b3_u5_cn3b)
				h_b3_u6_cn3b = tf.nn.relu(h_b3_u6_cn3b)


			# Block 4, unit 1
			with tf.name_scope('block4_unit1b'):

				# Original way to go on a resnet50
				# Calculating the first shortcut
				# shortcut_b4b = self.conv2ds2(h_b3_u6_cn3b, self.weight["W_b4_u1_cn0"])
				# shortcut_b4b = tf.layers.batch_normalization(inputs=shortcut_b4b , axis=3, momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True, scale=True, training=self.training, fused=True)

				# h_b4_u1_cn1b = self.conv2ds2(h_b3_u6_cn3b, self.weight["W_b4_u1_cn1"])
				# h_b4_u1_cn1b = tf.layers.batch_normalization(inputs=h_b4_u1_cn1b , axis=3, momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True, scale=True, training=self.training, fused=True)
				# h_b4_u1_cn1b = tf.nn.relu(h_b4_u1_cn1b)

				# Modification in the resnet 50 due to excesive reduction of the input data through the blocks
				# The modification is to use the conv2d function insted of the conv2ds2

				# Calculating the first shortcut
				shortcut_b4b = self.conv2d(h_b3_u6_cn3b, self.weight["W_b4_u1_cn0"])
				shortcut_b4b = tf.layers.batch_normalization(inputs=shortcut_b4b , axis=3, momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True, scale=True, training=self.training, fused=True)

				h_b4_u1_cn1b = self.conv2d(h_b3_u6_cn3b, self.weight["W_b4_u1_cn1"])
				h_b4_u1_cn1b = tf.layers.batch_normalization(inputs=h_b4_u1_cn1b , axis=3, momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True, scale=True, training=self.training, fused=True)
				h_b4_u1_cn1b = tf.nn.relu(h_b4_u1_cn1b)

				h_b4_u1_cn2b = self.conv2d(h_b4_u1_cn1b, self.weight["W_b4_u1_cn2"])
				h_b4_u1_cn2b = tf.layers.batch_normalization(inputs=h_b4_u1_cn2b , axis=3, momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True, scale=True, training=self.training, fused=True)
				h_b4_u1_cn2b = tf.nn.relu(h_b4_u1_cn2b)

				h_b4_u1_cn3b = self.conv2d(h_b4_u1_cn2b, self.weight["W_b4_u1_cn3"])
				h_b4_u1_cn3b = tf.layers.batch_normalization(inputs=h_b4_u1_cn3b , axis=3, momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True, scale=True, training=self.training, fused=True)
				h_b4_u1_cn3b = tf.add(h_b4_u1_cn3b, shortcut_b4b)
				h_b4_u1_cn3b = tf.nn.relu(h_b4_u1_cn3b)


			# Block 4, unit 2
			with tf.name_scope('block4_unit2b'):

				h_b4_u2_cn1b = self.conv2d(h_b4_u1_cn3b, self.weight["W_b4_u2_cn1"])
				h_b4_u2_cn1b = tf.layers.batch_normalization(inputs=h_b4_u2_cn1b , axis=3, momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True, scale=True, training=self.training, fused=True)
				h_b4_u2_cn1b = tf.nn.relu(h_b4_u2_cn1b)

				h_b4_u2_cn2b = self.conv2d(h_b4_u2_cn1b, self.weight["W_b4_u2_cn2"])
				h_b4_u2_cn2b = tf.layers.batch_normalization(inputs=h_b4_u2_cn2b , axis=3, momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True, scale=True, training=self.training, fused=True)
				h_b4_u2_cn2b = tf.nn.relu(h_b4_u2_cn2b)

				h_b4_u2_cn3b = self.conv2d(h_b4_u2_cn2b, self.weight["W_b4_u2_cn3"])
				h_b4_u2_cn3b = tf.layers.batch_normalization(inputs=h_b4_u2_cn3b , axis=3, momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True, scale=True, training=self.training, fused=True)
				h_b4_u2_cn3b = tf.add(h_b4_u2_cn3b, h_b4_u1_cn3b)
				h_b4_u2_cn3b = tf.nn.relu(h_b4_u2_cn3b)


			# Block 4, unit 3
			with tf.name_scope('block4_unit3b'):

				h_b4_u3_cn1b = self.conv2d(h_b4_u2_cn3b, self.weight["W_b4_u3_cn1"])
				h_b4_u3_cn1b = tf.layers.batch_normalization(inputs=h_b4_u3_cn1b , axis=3, momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True, scale=True, training=self.training, fused=True)
				h_b4_u3_cn1b = tf.nn.relu(h_b4_u3_cn1b)

				h_b4_u3_cn2b = self.conv2d(h_b4_u3_cn1b, self.weight["W_b4_u3_cn2"])
				h_b4_u3_cn2b = tf.layers.batch_normalization(inputs=h_b4_u3_cn2b , axis=3, momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True, scale=True, training=self.training, fused=True)
				h_b4_u3_cn2b = tf.nn.relu(h_b4_u3_cn2b)

				h_b4_u3_cn3b = self.conv2d(h_b4_u3_cn2b, self.weight["W_b4_u3_cn3"])
				h_b4_u3_cn3b = tf.layers.batch_normalization(inputs=h_b4_u3_cn3b , axis=3, momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True, scale=True, training=self.training, fused=True)
				h_b4_u3_cn3b = tf.add(h_b4_u3_cn3b, h_b4_u2_cn3b)
				h_b4_u3_cn3b = tf.nn.relu(h_b4_u3_cn3b)

			with tf.name_scope('pool1b'):
				h_pool2b = self.avg_pool_2x2(h_b4_u3_cn3b)


			# Fully connected
			with tf.name_scope('fc1'):
				h_concat = tf.concat([h_pool2a, h_pool2b], axis=3)
				h_concat_flat = tf.reshape(h_concat, [-1,2 * 2048 * self.height_after_conv * self.width_after_conv])
				Y_fc1 = tf.nn.relu(tf.matmul(h_concat_flat, self.weight["W_fc1"]))

				self.Y_logt = tf.matmul(Y_fc1, self.weight["W_fc2"])


class ModelBuilder():
	def __init__(self,		 
		in_height,
		in_width,
		channels,
		label,
		pool_layers):

		self.V7 = "V7"
		self.VGG11 = "VGG11"
		self.VGG13 = "VGG13"
		self.VGG16 = "VGG16"
		self.R50 = "R50"

		self.in_height = in_height
		self.in_width = in_width
		self.channels = channels
		self.label = label
		self.pool_layers = pool_layers

	def BuildType(self,model_type):
		if model_type == self.V7:
			return V7(self.in_height,self.in_width,self.channels,self.label,self.pool_layers)
		elif model_type == self.VGG11:
			return VGG11(self.in_height,self.in_width,self.channels,self.label,self.pool_layers)
		elif model_type == self.VGG13:
			return VGG13(self.in_height,self.in_width,self.channels,self.label,self.pool_layers)
		elif model_type == self.VGG16:
			return VGG16(self.in_height,self.in_width,self.channels,self.label,self.pool_layers)
		elif model_type == self.R50:
			return R50(self.in_height,self.in_width,self.channels,self.label,self.pool_layers)


class TFRecordReader():
	def __init__(self,
		train_tf_records_files = None,
        validation_tf_records_files = None,
        test_tf_records_files = None,
        parse_function = None,
        batch_size = None):

		self.train_tf_records_files = train_tf_records_files
		self.validation_tf_records_files = validation_tf_records_files
		self.test_tf_records_files = test_tf_records_files
		self.parse_function = parse_function
		self.batch_size = batch_size

		# ====================================
		# Training Dataset
		# ====================================
		train_dataset = tf.data.TFRecordDataset(self.train_tf_records_files)
		
		# Parse the record into tensors.
		train_dataset = train_dataset.map(self.parse_function)
		train_dataset = train_dataset.shuffle(buffer_size=10000)
		train_dataset = train_dataset.batch(self.batch_size)

		# ====================================
		# Validation Dataset
		# ====================================
		validation_dataset = tf.data.TFRecordDataset(self.validation_tf_records_files)

		# Parse the record into tensors.
		validation_dataset = validation_dataset.map(self.parse_function)
		validation_dataset = validation_dataset.shuffle(buffer_size=10000)
		validation_dataset = validation_dataset.batch(self.batch_size)

		# ====================================
		# test Dataset
		# ====================================
		test_dataset = tf.data.TFRecordDataset(self.test_tf_records_files)

		# Parse the record into tensors.
		test_dataset = test_dataset.map(self.parse_function)
		test_dataset = test_dataset.batch(self.batch_size)

		# ====================================
		# Defining the handler
		# ====================================
		self.train_handle = tf.placeholder(tf.string, shape=[])
		self.validation_handle = tf.placeholder(tf.string, shape=[])
		self.test_handle = tf.placeholder(tf.string, shape=[])
		
		self.train_iterator = tf.data.Iterator.from_string_handle(self.train_handle, train_dataset.output_types, train_dataset.output_shapes)
		self.validation_iterator = tf.data.Iterator.from_string_handle(self.validation_handle, validation_dataset.output_types, validation_dataset.output_shapes)
		self.test_iterator = tf.data.Iterator.from_string_handle(self.test_handle, test_dataset.output_types, test_dataset.output_shapes)

		self.train_next_element = self.train_iterator.get_next()
		self.validation_next_element = self.validation_iterator.get_next()
		self.test_next_element = self.test_iterator.get_next()
		
		# Defining the iterators
		self.training_iterator = train_dataset.make_initializable_iterator()
		self.validation_iterator = validation_dataset.make_initializable_iterator()
		self.test_iterator = test_dataset.make_initializable_iterator()


class Parser():
	def __init__(self,
		in_height,
		in_width,
		label,
		):

		self.in_height = in_height
		self.in_width = in_width
		self.label = label

	def parse_function(self,proto):

		# Defining the features to be loaded from the tfrecords file
		features = tf.parse_single_example(proto,
				# Defaults are not specified since both keys are required.
				features={
						'audio1': tf.FixedLenFeature([], tf.string),
						'audio2': tf.FixedLenFeature([], tf.string),
						'label': tf.FixedLenFeature([], tf.string),
				})

		# Convert from a scalar string tensor to a float32 tensor
		audio1= tf.decode_raw(features['audio1'], tf.float32)
		audio1 = tf.reshape(audio1,(self.in_height,self.in_width,1)) # If we want a flat vector #image.set_shape([in_heigth_size*in_with_size])
		audio1 = tf.cast(audio1, tf.float32)

		# Loading the second image as the first was loaded
		audio2= tf.decode_raw(features['audio2'], tf.float32)
		audio2 = tf.reshape(audio2,(self.in_height,self.in_width,1))
		audio2 = tf.cast(audio2, tf.float32)

		# Loading the labels 
		label= tf.decode_raw(features['label'], tf.int64)
		label.set_shape([self.label])

		return audio1, audio2, label

def file_observer(path, num_files):
	# Adquiring the data for the database
	database = np.array(glob.glob( os.path.join(path, '*.tfrecords') ))

	while database.shape[0] < num_files:
		time.sleep(1)
		database = np.array(glob.glob( os.path.join(path, '*.tfrecords') ))

	return

def plot_ROC (y_real, y_pred):
	# Plotting the ROC and dalculating the EER
	fpr, tpr, threshold = roc_curve(y_real, y_pred)

	plt.figure()
	plt.plot(fpr, tpr, color='darkorange', label='ROC curve')
	plt.xlim([-0.1, 1.0])
	plt.ylim([0.0, 1.05])
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('Receiver operating characteristic example')
	plt.savefig('ROC.png')
	plt.close()

	fnr = 1 - tpr
	eer_threshold = threshold[np.nanargmin(np.absolute((fnr - fpr)))]
	EER = fpr[np.nanargmin(np.absolute((fnr - fpr)))]

	return EER;

# ======================================================
# Loading the configuration for the model
# ======================================================
configuration_file = str(sys.argv[1])
if configuration_file == "":
    print("ERROR: you need to define param: config_model_datatype.json ")
    exit(0)

PARAMS = None

with open(configuration_file, 'r') as f:
    f = f.read()
    PARAMS = json.loads(f, object_hook=lambda d: namedtuple('X', d.keys())(*d.values()))

# Doing the process required for the loaded data
DIRECTORY_TRAIN = PARAMS.PATHS.directory_train
DIRECTORY_VALID = PARAMS.PATHS.directory_valid
DIRECTORY_TEST = PARAMS.PATHS.directory_test
DIRECTORY_WEIGHTS = PARAMS.PATHS.path_weights

N_ROW_TF_RECORD = PARAMS.DATA_GENERATOR.n_row_tf_record
ROUNDS_TRAIN = PARAMS.DATA_GENERATOR.rounds_train
ROUNDS_VALID = PARAMS.DATA_GENERATOR.rounds_valid
ROUNDS_TEST = PARAMS.DATA_GENERATOR.rounds_test
NUM_TRAIN_FILES = PARAMS.DATA_GENERATOR.num_train_files
NUM_VALID_FILES = PARAMS.DATA_GENERATOR.num_valid_files
NUM_TEST_FILES = PARAMS.DATA_GENERATOR.num_test_files

WINDOW = float(PARAMS.DATA_GENERATOR.window*PARAMS.DATA_GENERATOR.sample_rate)
MS = 1.0/PARAMS.DATA_GENERATOR.sample_rate
NPERSEG = int(PARAMS.DATA_GENERATOR.nperseg/MS)
NOVERLAP = int(PARAMS.DATA_GENERATOR.noverlap/MS)
NFFT = PARAMS.DATA_GENERATOR.nfft
DATA_TYPE = PARAMS.DATA_GENERATOR.data_type
SIZE_TIME = int((WINDOW - NPERSEG)/(NPERSEG - NOVERLAP))+1

if DATA_TYPE == "EmphSpec":
	IN_WIDTH = NFFT
	IN_HEIGHT = SIZE_TIME

elif DATA_TYPE == "Spec32":
	IN_WIDTH = SIZE_TIME
	IN_HEIGHT = 32

elif DATA_TYPE == "Spec256":
	IN_WIDTH = SIZE_TIME
	IN_HEIGHT = 256

elif DATA_TYPE == "Spec512":
	IN_WIDTH = SIZE_TIME
	IN_HEIGHT = 512

BATCH_SIZE = PARAMS.TRAINING.batch_size
NUM_EPOCHS = PARAMS.TRAINING.num_epochs
LEARNING_RATE = PARAMS.TRAINING.learning_rate
TYPE_OPT = PARAMS.TRAINING.type_opt
ARCHITECTURE = PARAMS.TRAINING.architecture
POOL_LAYERS = PARAMS.TRAINING.pool_layers
CHANNELS= PARAMS.TRAINING.channels
LABEL = PARAMS.TRAINING.label
RESTORE_WEIGHTS = PARAMS.TRAINING.restore_weights

_BATCH_NORM_DECAY = 0.997
_BATCH_NORM_EPSILON = 1e-5

NAME_OUTPUT = ARCHITECTURE + '_' + DATA_TYPE + '_' + TYPE_OPT + '_' + str(LEARNING_RATE) + '_' + str(NUM_EPOCHS)
OUTPUT_FOLDER = NAME_OUTPUT
OUTPUT_FILE = open(NAME_OUTPUT+'_results.txt', 'w')

if os.path.exists(OUTPUT_FOLDER) == False:
	os.mkdir(OUTPUT_FOLDER)

DELETE_FILE_EVERY = int(N_ROW_TF_RECORD/BATCH_SIZE)

# ======================================================
# Creating the network model
# ======================================================
builder = ModelBuilder(in_height = IN_HEIGHT,
		in_width = IN_WIDTH,
		channels = CHANNELS,
		label = LABEL,
		pool_layers = POOL_LAYERS)

network = builder.BuildType(ARCHITECTURE)

if TYPE_OPT == "grad":
	optimizer = tf.train.GradientDescentOptimizer(learning_rate=LEARNING_RATE).minimize(network.loss)

# ======================================================
# Creating the parser 
# ======================================================
parser = Parser(in_height = IN_HEIGHT,
		in_width = IN_WIDTH,
		label = LABEL)

# ======================================================
# Creating the tf records reader
# ======================================================
train_file = []
validation_file = []
test_file = []

for i in range(0,NUM_TRAIN_FILES):
	train_file.append(DIRECTORY_TRAIN+'/'+ 'train_' +str(i) + '.tfrecords')

for i in range(0,NUM_VALID_FILES):
	validation_file.append(DIRECTORY_VALID+'/'+ 'validation_' +str(i) + '.tfrecords')

for i in range(0,NUM_TEST_FILES):
	test_file.append(DIRECTORY_TEST+'/'+ 'test_' +str(i) + '.tfrecords')

tf_records_reader = TFRecordReader(train_tf_records_files = train_file,
        validation_tf_records_files = validation_file,
        test_tf_records_files = test_file,
        parse_function = parser.parse_function,
        batch_size = BATCH_SIZE)

# ======================================================
# Running the model
# ======================================================

# The op for initializing the variables.
init_op = tf.group(tf.global_variables_initializer(),tf.local_variables_initializer())

# opens session
with tf.Session() as sess:
	
	# writers for TensorBorad
	train_writer = tf.summary.FileWriter('graphs/train_'+ NAME_OUTPUT)
	valid_writer = tf.summary.FileWriter('graphs/valid_'+ NAME_OUTPUT)
	test_writer = tf.summary.FileWriter('graphs/test_'+ NAME_OUTPUT)
	train_writer.add_graph(sess.graph)

	# initialize variables (params)
	sess.run(init_op)
	training_handle = sess.run(tf_records_reader.training_iterator.string_handle())
	validation_handle = sess.run(tf_records_reader.validation_iterator.string_handle())
	test_handle = sess.run(tf_records_reader.test_iterator.string_handle())

	saver = tf.train.Saver()
	if RESTORE_WEIGHTS == "True":
		print("Restoring weights")
		saver.restore(sess, DIRECTORY_WEIGHTS)
	elif RESTORE_WEIGHTS == "False":
		print("No weights restored")
	else:
		print("ERROR: you need to indicate if the weights should or shouldn't be restored.")

	# Initializing the step for train and validation
	step_train = 1
	step_valid = 1
	step_test = 1

	acc_train = 0
	acc_valid = 0
	acc_test = 0

	for n_epochs in xrange(NUM_EPOCHS):
		
		for n_rounds_train in xrange(ROUNDS_TRAIN):			
			n_file_train = 1
			file_observer(DIRECTORY_TRAIN, NUM_TRAIN_FILES)
			sess.run(tf_records_reader.training_iterator.initializer)

			# Running the training
			while True:

				try:
					# evaluation with train data
					X1_array, X2_array, Y_array = sess.run(tf_records_reader.train_next_element, feed_dict={tf_records_reader.train_handle: training_handle})
					feed_dict = {network.X1: X1_array, network.X2: X2_array, network.Y : Y_array, network.training:1}
					fetches = [optimizer, network.loss, network.accuracy, network.summary ]
					_,train_loss, train_acc, train_summary = sess.run(fetches, feed_dict=feed_dict)
					train_writer.add_summary(train_summary, step_train)

					acc_train = acc_train + train_acc

					# Printing the results every 100 batch
					if step_train % 100 == 0:
						msg = "I{:3d} loss_train: ({:6.8f}), acc_train(batch, global): ({:6.8f},{:6.8f})"
						msg = msg.format(step_train, train_loss, train_acc/BATCH_SIZE, acc_train/(BATCH_SIZE*step_train))
						print(msg)
						OUTPUT_FILE.write(msg + '\n')

					if (n_file_train)%DELETE_FILE_EVERY == 0:
						number_file = int((n_file_train/DELETE_FILE_EVERY)-1)
						os.remove(train_file[number_file])

					step_train += 1		 
					n_file_train+=1

				except tf.errors.OutOfRangeError:
					# If the data ended the while must be broken
					break;

		print('End training epoch')
		# Saving the weightsin every epoch
		save_path = saver.save(sess, str(OUTPUT_FOLDER+'/'+ str(n_epochs) +'weights.ckpt') )

		for n_rounds_valid in xrange(ROUNDS_VALID):
			
			n_file_valid = 1
			file_observer(DIRECTORY_VALID, NUM_VALID_FILES)
			sess.run(tf_records_reader.validation_iterator.initializer)

			# Running the validation
			while True:

				try:
					# evaluation with valid data
					X1_array, X2_array, Y_array = sess.run(tf_records_reader.validation_next_element, feed_dict={tf_records_reader.validation_handle: validation_handle})
					feed_dict = {network.X1: X1_array, network.X2: X2_array, network.Y : Y_array, network.training:0}
					fetches = [network.loss, network.accuracy, network.summary ]
					valid_loss, valid_acc, valid_summary = sess.run(fetches, feed_dict=feed_dict)
					valid_writer.add_summary(valid_summary, step_train)

					acc_valid = acc_valid + valid_acc

					# Printing the results every 100 batch
					if step_valid % 100 == 0:
						msg = "I{:3d} loss_valid: ({:6.8f}), acc_valid(batch, global): ({:6.8f},{:6.8f})"
						msg = msg.format(step_valid, valid_loss, valid_acc/BATCH_SIZE, acc_valid/(BATCH_SIZE*step_valid))
						print(msg)
						OUTPUT_FILE.write(msg + '\n')

					if (n_file_valid)%DELETE_FILE_EVERY == 0:
						number_file = int((n_file_valid/DELETE_FILE_EVERY)-1)
						os.remove(validation_file[number_file])

					step_valid += 1		 
					n_file_valid+=1

				except tf.errors.OutOfRangeError:
					# If the data ended the while must be broken
					break;

	y_real =[]
	y_pred = []

	for n_rounds_test in xrange(ROUNDS_TEST):
			
		n_file_test = 1
		file_observer(DIRECTORY_TEST, NUM_TEST_FILES)
		sess.run(tf_records_reader.test_iterator.initializer)

		# Running the test
		while True:

			try:
				# evaluation with test data
				X1_array, X2_array, Y_array = sess.run(tf_records_reader.test_next_element, feed_dict={tf_records_reader.test_handle: test_handle})
				feed_dict = {network.X1: X1_array, network.X2: X2_array, network.Y : Y_array, network.training:0}
				fetches = [network.loss, network.accuracy, network.summary, network.Y_pred ]
				test_loss, test_acc, test_summary, Y_pred = sess.run(fetches, feed_dict=feed_dict)
				test_writer.add_summary(test_summary, step_train)

				acc_test = acc_test + test_acc

				# Printing the results every 100 batch
				if step_test % 100 == 0:
					msg = "I{:3d} loss_test: ({:6.8f}), acc_test(batch, global): ({:6.8f},{:6.8f})"
					msg = msg.format(step_test, test_loss, test_acc/BATCH_SIZE, acc_test/(BATCH_SIZE*step_test))
					print(msg)
					OUTPUT_FILE.write(msg + '\n')

				if (n_file_test)%DELETE_FILE_EVERY == 0:
					number_file = int((n_file_test/DELETE_FILE_EVERY)-1)
					os.remove(test_file[number_file])

				step_test += 1		 
				n_file_test+=1

				for index in xrange(0,Y_array.shape[0]):

					y_real.append(Y_array[index, 1])
					y_pred.append(Y_pred[index, 1])

			except tf.errors.OutOfRangeError:
				# If the data ended the while must be broken
				break;

	EER = plot_ROC (y_real, y_pred)
	msg = 'EER: ' + str(EER)
	print(msg)
	OUTPUT_FILE.write(msg + '\n')