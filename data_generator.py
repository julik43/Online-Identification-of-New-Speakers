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

from scipy.signal import lfilter, hamming
from scipy.fftpack import fft
from scikits.talkbox import segment_axis

import os.path
import os
import sys
import glob
import json
import numpy as np
import random
import soundfile as sf
import tensorflow as tf

from scipy import signal
from sklearn.preprocessing import StandardScaler
from collections import namedtuple

# silences Tensorflow boot logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Using just one GPU in case of GPU 
# os.environ['CUDA_VISIBLE_DEVICES']= '0'

# ======================================================
# Functions and methods need it
# ======================================================
def get_part(part,string):
	# Function that splits and string to get the desire part
	aux = string.split('/')
	a = aux[len(aux)-part-1]
	return a

def get_class(direction):
	# Getting the class of the audio
	fixed_class = get_part(2,direction)
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

# Functions to write the data in TFRecord format
def _int64_feature(value):
	return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
	return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def convert_to(audio1, audio2, labels, name):
	"""Converts a dataset to tfrecords."""
	num_examples = audio1.shape[0]
	rows = audio1.shape[1]
	cols = audio1.shape[2]
	depth = audio1.shape[3]
	filename = os.path.join(name + '.tfrecords')

	print('Writing', filename)
	writer = tf.python_io.TFRecordWriter(filename)
	for index in range(num_examples):

		audio1_raw = audio1[index].tostring()
		audio2_raw = audio2[index].tostring()
		label = labels[index].tostring()
		
		example = tf.train.Example(features=tf.train.Features(feature={
				'height': _int64_feature(rows),
				'width': _int64_feature(cols),
				'depth': _int64_feature(depth),
				'audio1': _bytes_feature(audio1_raw),
				'audio2': _bytes_feature(audio2_raw),
				'label': _bytes_feature(label)}))
		writer.write(example.SerializeToString())
	writer.close()

def extra_params(section, step_tf):

	data_start = []
	data_end = []

	if section == 0:
		path = DIRECTORY_TRAIN
		name_base = (path+'/train')
		total_files = int(NUM_TRAIN_FILES*ROUNDS_TRAIN/step_tf)
		input_file	= open(FILE_TRAIN,'r')
		num_total_files_per_round = NUM_TRAIN_FILES
		if DATA_VAD_TRAIN_START != "":
			data_start = np.load(DATA_VAD_TRAIN_START)
			data_end = np.load(DATA_VAD_TRAIN_END)

	elif section == 1:			
		path = DIRECTORY_VALID
		name_base = (path+'/validation')
		total_files = int(NUM_VALID_FILES*ROUNDS_VALID/step_tf)
		input_file	= open(FILE_VALID,'r')
		num_total_files_per_round = NUM_VALID_FILES
		if DATA_VAD_VALID_START != "":
			data_start = np.load(DATA_VAD_VALID_START)
			data_end = np.load(DATA_VAD_VALID_END)

	elif section == 2 :
		path = DIRECTORY_TEST
		name_base = (path+'/test')
		total_files = int(NUM_TEST_FILES*ROUNDS_TEST/step_tf)
		input_file	= open(FILE_TEST,'r')
		num_total_files_per_round = NUM_TEST_FILES
		if DATA_VAD_TEST_START != "":
			data_start = np.load(DATA_VAD_TEST_START)
			data_end = np.load(DATA_VAD_TEST_END)

	return path, name_base, total_files, input_file, num_total_files_per_round, data_start, data_end	

def read_file(file):
	matrix = []
	for line in file:
		row = line.rstrip()
		class_row = str(get_class(row))
		matrix.append([row, class_row])
	return matrix

def create_data_array(matrix, row_matrix, data_start, data_end):

	X1 = []
	X2 = []
	Y = []
	rows = len(matrix)

	while len(X1)< N_ROW_TF_RECORD:
		
		chosen_audio_1 = matrix[row_matrix][0]
		fixed_class = matrix[row_matrix][1]
		audio_1,samplerate = sf.read(chosen_audio_1)

		if data_start != []:
			start = data_start[row_matrix]
			end = data_end[row_matrix]
		else:
			start,end = vad_analysis(audio_1, samplerate)

		if start.shape[0]>0:		

			# Listing all the audios of the same class
			list_same_class = []
			for i_same_class in xrange(0,len(matrix)):
				if matrix[i_same_class][1] == fixed_class:
					list_same_class.append(i_same_class)

			while True:
				
				# Chosing randomly an audio of the same kind
				r_index = random.randrange(0,len(list_same_class),1)
				row_random = list_same_class[r_index]
				chosen_audio_2 = matrix[row_random][0]
				audio_2,samplerate = sf.read(chosen_audio_2)

				if data_start != []:
					start2 = data_start[row_random]
					end2 = data_end[row_random]
				else:
					start2,end2 = vad_analysis(audio_2, samplerate)

				if start2.shape[0]>0:			
					
					data_audio_1= generate_data(DATA_TYPE, audio_1, start, end, samplerate)
					data_audio_2= generate_data(DATA_TYPE, audio_2, start2, end2, samplerate)

					X1.append(data_audio_1)
					X2.append(data_audio_2)
					Y.append([0,1])
					break

			while True:
				r_index = random.randrange(0,rows,1)
				chosen_audio_2 = matrix[r_index][0]

				if matrix[r_index][1]!=fixed_class:
					
					audio_2,samplerate = sf.read(chosen_audio_2)

					if data_start != []:
						start2 = data_start[r_index]
						end2 = data_end[r_index]
					else:
						start2,end2 = vad_analysis(audio_2, samplerate)

					if start2.shape[0]>0:					
					
						data_audio_1= generate_data(DATA_TYPE, audio_1, start, end, samplerate)
						data_audio_2= generate_data(DATA_TYPE, audio_2, start2, end2, samplerate)

						# Filling the matrixes with the data
						X1.append(data_audio_1)
						X2.append(data_audio_2)
						Y.append([1,0])
						break

		row_matrix+= step_tf

		if row_matrix >= rows:
			row_matrix = initial_number_tf

	return X1, X2, Y, row_matrix


# ======================================================
# Loading the configuration for the model
# ======================================================
configuration_file = str(sys.argv[1])
if configuration_file == "":
    print("ERROR: you need to define param: config_model_datatype.json ")
    exit(0)
    
initial_number_tf = int(sys.argv[2])
step_tf = int(sys.argv[3])

if initial_number_tf == "":
    print("ERROR: you need to define the initial number of the TFRecord file (recommend: python data_generator 0 1)")
    exit(0)

if step_tf == "":
    print("ERROR: you need to define the step for the TFRecord file (recommend: python data_generator 0 1)")
    exit(0)

PARAMS = None

with open(configuration_file, 'r') as f:
    f = f.read()
    PARAMS = json.loads(f, object_hook=lambda d: namedtuple('X', d.keys())(*d.values()))

# Doing the process required for the loaded data
DIRECTORY_TRAIN = PARAMS.PATHS.directory_train
DIRECTORY_VALID = PARAMS.PATHS.directory_valid
DIRECTORY_TEST = PARAMS.PATHS.directory_test
FILE_TRAIN = PARAMS.PATHS.file_train
FILE_VALID = PARAMS.PATHS.file_valid
FILE_TEST = PARAMS.PATHS.file_test
DATA_VAD_TRAIN_START = PARAMS.PATHS.data_vad_train_start
DATA_VAD_TRAIN_END = PARAMS.PATHS.data_vad_train_end
DATA_VAD_VALID_START = PARAMS.PATHS.data_vad_valid_start
DATA_VAD_VALID_END = PARAMS.PATHS.data_vad_valid_end
DATA_VAD_TEST_START = PARAMS.PATHS.data_vad_test_start
DATA_VAD_TEST_END = PARAMS.PATHS.data_vad_test_end

N_ROW_TF_RECORD = PARAMS.DATA_GENERATOR.n_row_tf_record
ROUNDS_TRAIN = PARAMS.DATA_GENERATOR.rounds_train
ROUNDS_VALID = PARAMS.DATA_GENERATOR.rounds_valid
ROUNDS_TEST = PARAMS.DATA_GENERATOR.rounds_test
NUM_TRAIN_FILES = PARAMS.DATA_GENERATOR.num_train_files
NUM_VALID_FILES = PARAMS.DATA_GENERATOR.num_valid_files
NUM_TEST_FILES = PARAMS.DATA_GENERATOR.num_test_files

WINDOW = int(PARAMS.DATA_GENERATOR.window*PARAMS.DATA_GENERATOR.sample_rate)
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

NUM_EPOCHS = PARAMS.TRAINING.num_epochs

# Variables for VAD analysis
NFFT_VAD = 512
LOW_BAN = 300
HIGHT_BAN = 3000

# ======================================================
# Creating the data generator
# ======================================================
if os.path.exists(str(DIRECTORY_TRAIN)) == False:
	os.mkdir(str(DIRECTORY_TRAIN))

if os.path.exists(str(DIRECTORY_VALID)) == False:
	os.mkdir(str(DIRECTORY_VALID))

if os.path.exists(str(DIRECTORY_TEST)) == False:
	os.mkdir(str(DIRECTORY_TEST))

# Compute for the desired number of epochs.
for n_epochs in range(NUM_EPOCHS):

	limit = 3 if n_epochs == (NUM_EPOCHS-1) else 2

	for section in range(0,limit):

		path, name_base, total_files, input_file, num_total_files_per_round, data_start, data_end = extra_params(section, step_tf)		
		matrix = read_file(input_file)

		row_matrix = initial_number_tf
		num_files = initial_number_tf

		for n_file in xrange(total_files):	

			X1, X2, Y, row_matrix = create_data_array(matrix, row_matrix, data_start, data_end)

			name = name_base + '_'+str(num_files)

			X1_array = np.array(X1, dtype = np.float32)
			X2_array = np.array(X2, dtype = np.float32)
			Y_array = np.array(Y)

			permutation = np.random.permutation(X1_array.shape[0])
			X1_array = X1_array[permutation,:]
			X2_array = X2_array[permutation,:]
			Y_array = Y_array[permutation]

			print(X1_array.shape)
			print(X2_array.shape)
			print(Y_array.shape)
			print(len(X1_array))

			# Veryfing that the data to write is not in use
			database = np.array(glob.glob( os.path.join(path, '*.tfrecords') ))
			exists = np.where(database==(name + '.tfrecords'))

			while len(exists[0]) > 0:
				database = np.array(glob.glob( os.path.join(path, '*.tfrecords') ))
				exists = np.where(database==(name + '.tfrecords'))

			convert_to(X1_array,X2_array,Y_array,name)

			num_files += step_tf							

			if num_files >= num_total_files_per_round:
				num_files = initial_number_tf