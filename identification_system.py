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
import sounddevice as sd
import random

from sklearn.metrics import roc_curve
from collections import namedtuple
# from tensorflow.contrib import layers
from scipy import signal
from sklearn.preprocessing import StandardScaler
from scipy.signal import lfilter, hamming
from scipy.fftpack import fft
from scikits.talkbox import segment_axis

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


# ======================================================
# Functions and methods to generate the data
# ======================================================
def get_part(part,string):
	# Function that splits and string to get the desire part
	aux = string.split('/')
	a = aux[len(aux)-part-1]
	return a

def get_class(direction):
	# Getting the class of the audio
	class_index = direction.rfind('/')
	fixed_class = direction[class_index+1:len(direction)]

	class_index = fixed_class.rfind('_')
	if class_index >= 0:
		fixed_class = fixed_class[0:class_index]

	return fixed_class

def pre_proccessing(audio, rate, pre_emphasis = 0.97, frame_size=0.02, frame_stride=0.01):
	emphasized_audio = np.append(audio[0], audio[1:] - pre_emphasis * audio[:-1])
	frame_length, frame_step = frame_size * rate, frame_stride * rate	# Convert from seconds to samples
	audio_length = len(emphasized_audio) 
	frame_length = int(round(frame_length))
	frame_step = int(round(frame_step))
	num_frames = int(np.ceil(float(np.abs(audio_length - frame_length)) / frame_step))	# Make sure that we have at least 1 frame
	pad_audio_length = num_frames * frame_step + frame_length
	z = np.zeros((pad_audio_length - audio_length))
	pad_audio = np.append(emphasized_audio, z) # Pad audio to make sure that all frames have equal number of samples without truncating any samples from the original audio
	indices = np.tile(np.arange(0, frame_length), (num_frames, 1)) + np.tile(np.arange(0, num_frames * frame_step, frame_step)\
	, (frame_length, 1)).T
	frames = pad_audio[indices.astype(np.int32, copy=False)]
	return frames

def power_spect(audio, rate):
	frames = pre_proccessing(audio, rate)
	mag_frames = np.absolute(np.fft.rfft(frames, NFFT_VAD))	# Magnitude of the FFT
	pow_frames = ((1.0 / NFFT_VAD) * ((mag_frames) ** 2))	# Power Spectrum
	return pow_frames

def mel_filter(audio, rate, nfilt = 40):
	pow_frames = power_spect(audio, rate)
	low_freq_mel = 0
	high_freq_mel = (2595 * np.log10(1 + (rate / 2) / 700))	# Convert Hz to Mel
	mel_points = np.linspace(low_freq_mel, high_freq_mel, nfilt + 2)	# Equally spaced in Mel scale
	hz_points = (700 * (10**(mel_points / 2595) - 1))	# Convert Mel to Hz
	bin = np.floor((NFFT_VAD + 1) * hz_points / rate)
	fbank = np.zeros((nfilt, int(np.floor(NFFT_VAD / 2 + 1))))

	for m in range(1, nfilt + 1):
		 f_m_minus = int(bin[m - 1])	 # left
		 f_m = int(bin[m])						 # center
		 f_m_plus = int(bin[m + 1])		# right

		 for k in range(f_m_minus, f_m):
				fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
		 for k in range(f_m, f_m_plus):
				fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])
	

	filter_banks = np.dot(pow_frames, fbank.T)
	filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)	# Numerical Stability
	return hz_points ,filter_banks

def voice_frecuency(audio,rate):
	frec_wanted = []
	hz_points, filter_banks = mel_filter(audio, rate)
	for i in range(len(hz_points)-2):
		 if hz_points[i]<= HIGHT_BAN and hz_points[i] >=LOW_BAN:
				frec_wanted.append(1)
		 else:
				frec_wanted.append(0)
	
	sum_voice_energy = np.dot(filter_banks, frec_wanted)/1e+6	## 1e+6 is use to reduce the audio amplitud 
	return(sum_voice_energy)

def get_points(aux, sr=16000, frame_size=0.02, frame_stride=0.01):
	flag_audio = False
	cont_silence = 0 
	init_audio = 0
	start =[]
	end = []
	min_frames = 40
	threshold = np.max(aux) * 0.04

	for i in range(len(aux)):
		if aux[i]	< threshold:
			cont_silence+=1

			if cont_silence == min_frames:
				if flag_audio == True:
					start.append(init_audio)
					end.append(i-min_frames+1)
					flag_audio = False
			
		if aux[i] > threshold:
			if flag_audio == False:
				init_audio = i
				flag_audio = True

			cont_silence=0

	if flag_audio == True:
		start.append(init_audio)
		end.append(len(aux))

	start = (np.array(start) * frame_stride * sr).astype(int)
	end = (np.array(end) * frame_stride * sr).astype(int)

	return start,end

def vad_analysis(audio, samplerate):
	# Analyzing the VAD of the audio
	voice_energy = voice_frecuency(audio, samplerate)
	start, end= get_points(voice_energy,samplerate)
	r_start = []
	r_end = []

	for i in xrange(0,start.shape[0]):
		if end[i] - start[i] > WINDOW:
			r_start.append(start[i])
			r_end.append(end[i])

	return np.array(r_start),np.array(r_end)

# Functions to generate the data 
def preemp(audio, p):
	"""Pre-emphasis filter."""
	return lfilter([1., -p], 1, audio)

def get_emph_spec(audio, nperseg=256, noverlap = 96, nfft=512, fs=16000):
	# Function to generate the emphasized spectrogram
	prefac = 0.97
	w = hamming(nperseg, sym=0)
	extract = preemp(audio, prefac)
	framed = segment_axis(extract, nperseg, noverlap) * w
	spec = np.abs(fft(framed, nfft, axis=-1))
	return spec

def generate_data(data_type, audio, start, end, samplerate = 16000):

	# Choosing randomly a window that fits the specifications
	option = random.randrange(0,start.shape[0],1)
	index = random.randrange(start[option],end[option]-WINDOW,1)
	audio_data = audio[index:index+WINDOW]

	if data_type == 'Spec32' or data_type == 'Spec256' or data_type == 'Spec512':
		f, t, Sxx = signal.spectrogram(audio_data, fs = samplerate,	window=('hamming'), nperseg=NPERSEG, noverlap=NOVERLAP, nfft=NFFT, detrend='constant', return_onesided=False, scaling='density', axis=-1)
		Hxx = StandardScaler().fit_transform(Sxx)
		data_audio = np.reshape(Hxx[0:IN_HEIGHT,:],(IN_HEIGHT,IN_WIDTH,1))
	
	elif data_type == 'EmphSpec':
		spec = get_emph_spec(audio_data, nperseg=NPERSEG, noverlap=NOVERLAP, nfft=NFFT,fs=samplerate)
		data_audio = np.reshape(spec[:,:],(IN_HEIGHT, IN_WIDTH,1))	

	return data_audio

def read_file(file):
	matrix = []
	for line in file:
		row = line.rstrip()
		matrix.append([row])
	return matrix

def load_db():

	database = []

	aux = glob.glob( os.path.join(DIRECTORY_DATABASE, '*.npy') )

	for i in xrange(0,len(aux)):
		row_class = get_class(aux[i])
		database.append([str(aux[i]), str(row_class)])

	return database

def create_data_array(matrix, row_matrix, data_audio_2):

	X1 = []
	X2 = []
	Y = []
	X1_class = []

	data_audio_1 = np.load(matrix[row_matrix][0])
	class_audio_1 = matrix[row_matrix][1]

	X1.append(data_audio_1)
	X2.append(data_audio_2)
	Y.append([0,0])
	X1_class.append(class_audio_1)

	row_matrix = row_matrix + 1	

	return np.array(X1), np.array(X2), np.array(Y), X1_class, row_matrix, data_audio_2


def statistics_audio(Y_Aux, total_ver_y_pred):
	unique_classes = np.unique(np.array(Y_Aux))
	class_value = np.zeros((len(unique_classes)))
	
	for number_class in range(0,len(unique_classes)):

		num_audios = 0

		for row_pred in xrange(0,len(total_ver_y_pred)):

		  if Y_Aux[row_pred] == unique_classes[number_class]:
			class_value[number_class] = class_value[number_class] + total_ver_y_pred[row_pred][1]
			num_audios+=1

		class_value[number_class] = class_value[number_class]/num_audios

	# Initializing the value of the class
	value_class = 'unknown'
	value_y_pred = 0

	# Choosing only the class with the highest score above 0.5
	for row_pred in xrange(0,len(class_value)):
	  
	  if class_value[row_pred] > 0.5 and class_value[row_pred] > value_y_pred:

		value_class = unique_classes[row_pred]
		value_y_pred = class_value[row_pred]	

	return value_class


def get_number_audio(direction):
	# Getting the class of the audio
	class_index = direction.rfind('/')
	fixed_class = direction[class_index+1:len(direction)]

	class_index = fixed_class.rfind('.')
	fixed_class = fixed_class[0:class_index]

	class_index = fixed_class.rfind('_')
	if class_index >= 0:
		number_audio = fixed_class[class_index+1:]

	return int(number_audio)

def save_data(database, data, name):

	number_audio = 0

	for i in xrange(0,len(database)):
		class_data = database[i][1]
		numb_audio = get_number_audio(database[i][0])

		if class_data == name and numb_audio > number_audio:
			number_audio = numb_audio

	number_audio+=1
	name_speaker = name + '_' + str(number_audio)
	np.save(DIRECTORY_DATABASE+'/'+str(name_speaker), data)





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
DIRECTORY_DATABASE = PARAMS.PATHS.database
DIRECTORY_WEIGHTS = PARAMS.PATHS.path_weights

SAMPLERATE = PARAMS.DATA_GENERATOR.sample_rate
WINDOW = int(PARAMS.DATA_GENERATOR.window*SAMPLERATE)
MS = 1.0/SAMPLERATE
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

ARCHITECTURE = PARAMS.TRAINING.architecture
POOL_LAYERS = PARAMS.TRAINING.pool_layers
CHANNELS= PARAMS.TRAINING.channels
LABEL = PARAMS.TRAINING.label
RESTORE_WEIGHTS = PARAMS.TRAINING.restore_weights

# Variables for VAD analysis
NFFT_VAD = 512
LOW_BAN = 300
HIGHT_BAN = 3000

_BATCH_NORM_DECAY = 0.997
_BATCH_NORM_EPSILON = 1e-5

NAME_OUTPUT = DIRECTORY_DATABASE
OUTPUT_FOLDER = NAME_OUTPUT

if os.path.exists(OUTPUT_FOLDER) == False:
	os.mkdir(OUTPUT_FOLDER)

# ======================================================
# Creating the network model
# ======================================================
builder = ModelBuilder(in_height = IN_HEIGHT,
		in_width = IN_WIDTH,
		channels = CHANNELS,
		label = LABEL,
		pool_layers = POOL_LAYERS)

network = builder.BuildType(ARCHITECTURE)


# ======================================================
# Running the model
# ======================================================

# The op for initializing the variables.
init_op = tf.group(tf.global_variables_initializer(),tf.local_variables_initializer())

# opens session
with tf.Session() as sess:
	
	# initialize variables (params)
	sess.run(init_op)
	
	saver = tf.train.Saver()
	if RESTORE_WEIGHTS == "True":
		print("Restoring weights")
		saver.restore(sess, DIRECTORY_WEIGHTS)
	elif RESTORE_WEIGHTS == "False":
		print("No weights restored")
		print("You need to restore the corresponding weights to continue.")
		exit(0)
	else:
		print("ERROR: you need to indicate if the weights should or shouldn't be restored.")
		exit(0)


	while True:

		os.system('clear') 
		print('Please press any character when you want to start speaking') 
		option = raw_input()
		audio_length = WINDOW + SAMPLERATE * 1
		recording = np.zeros(int(audio_length))
		recording = sd.rec(int(audio_length), samplerate=SAMPLERATE, channels=1)
		time.sleep(audio_length/SAMPLERATE) 
		print('Thanks, you can stop speaking')	

		start,end = vad_analysis(recording, SAMPLERATE)
		if len(start)>0:	   	

			database = load_db()
			row_matrix = 0
			total_ver_y_pred = []
			total_class = []

			data_audio_2= generate_data(DATA_TYPE, recording, start, end, SAMPLERATE)


			while row_matrix < len(database):			
				X1, X2, Y, X1_class, row_matrix, data_record = create_data_array(database, row_matrix, data_audio_2)
				feed_dict = {network.X1: X1, network.X2: X2, network.Y : Y, network.training:0}
				fetches = [network.Y_pred, network.label_pred]
				ver_y_pred,_ = sess.run(fetches, feed_dict=feed_dict)
				
				# Updating the total results
				total_ver_y_pred = total_ver_y_pred + ver_y_pred.tolist()
				total_class = total_class + X1_class

			name_speaker = statistics_audio(total_class, total_ver_y_pred)

			print("\nYou are", name_speaker, "\n" )

			speaker_correct = 'n'
			if name_speaker != 'unknown':
				print("Are you this speaker? (y--> yes, n-->no)")
				speaker_correct = raw_input()

			if speaker_correct == 'n':
				print('Please write your name: ')
				name_speaker = raw_input()			

			save_data(database, data_audio_2, name_speaker)

		else:
			print("Please speak louder and enough time.")
