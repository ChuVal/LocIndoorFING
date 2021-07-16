import os
import math
import numpy as np
import datetime as dt
import tensorflow as tf
import matplotlib.pyplot as plt
from numpy import newaxis
from core.utils import Timer
from tensorflow.keras.layers import Dense, Activation, Dropout, LSTM, TimeDistributed
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import plot_model


class Model():


	def __init__(self):
		self.model = Sequential()

	def load_model(self, filepath):
		print('[Model] Loading model from file %s' % filepath)
		self.model = load_model(filepath)

	def model_clas(self, configs):
		timer = Timer()
		timer.start()

		for layer in configs['model_clas']['layers']:
			neurons = layer['neurons'] if 'neurons' in layer else None
			dropout_rate = layer['rate'] if 'rate' in layer else None
			activation = layer['activation'] if 'activation' in layer else None
			return_seq = layer['return_seq'] if 'return_seq' in layer else None
			input_timesteps = layer['input_timesteps'] if 'input_timesteps' in layer else None
			input_dim = layer['input_dim'] if 'input_dim' in layer else None
			output_layer = layer['output_layer'] if 'output_layer' in layer else None

			if layer['type'] == 'dense':
				self.model.add(Dense(neurons, activation=activation))
			if layer['type'] == 'lstm':
				self.model.add(LSTM(neurons, input_shape=(input_timesteps, input_dim), return_sequences=return_seq))
			if layer['type'] == 'dropout':
				self.model.add(Dropout(dropout_rate))
			if layer['type'] == 'TimeDistributed':
				self.model.add(TimeDistributed(Dense(output_layer)))

		self.model.summary()
		self.model.compile(loss=configs['model_clas']['loss'], optimizer=configs['model_clas']['optimizer'], metrics=['acc'])

		print('[Model] Model Clasificador Compiled')
		timer.stop()

	def custom_loss(self, y_true, y_pred):
		(x1,y1) = y_true[0], y_true[1]
		(x2,y2) = y_pred[0], y_pred[1]
		loss = ((x2-x1)**2 + (y2-y1)**2)**(1/2)
		return loss

	def model_MIMO(self, configs):
		timer = Timer()
		timer.start()

		for layer in configs['model_mimo']['layers']:
			neurons = layer['neurons'] if 'neurons' in layer else None
			dropout_rate = layer['rate'] if 'rate' in layer else None
			activation = layer['activation'] if 'activation' in layer else None
			return_seq = layer['return_seq'] if 'return_seq' in layer else None
			input_timesteps = layer['input_timesteps'] if 'input_timesteps' in layer else None
			input_dim = layer['input_dim'] if 'input_dim' in layer else None
			output_layer = layer['output_layer'] if 'output_layer' in layer else None

			if layer['type'] == 'dense':
				self.model.add(Dense(neurons, activation=activation))
			if layer['type'] == 'lstm':
				self.model.add(LSTM(neurons, input_shape=(input_timesteps, input_dim), return_sequences=return_seq))
			if layer['type'] == 'dropout':
				self.model.add(Dropout(dropout_rate))
			if layer['type'] == 'TimeDistributed':
				self.model.add(TimeDistributed(Dense(output_layer)))

		self.model.summary()
		self.model.compile(loss=self.custom_loss, optimizer=configs['model_mimo']['optimizer'], metrics=['acc'])

		print('[Model] Model Mimo Compiled')
		timer.stop()
        
	def model_PMIMO(self, configs):
		timer = Timer()
		timer.start()

		for layer in configs['model_Pmimo']['layers']:
			neurons = layer['neurons'] if 'neurons' in layer else None
			dropout_rate = layer['rate'] if 'rate' in layer else None
			activation = layer['activation'] if 'activation' in layer else None
			return_seq = layer['return_seq'] if 'return_seq' in layer else None
			input_timesteps = layer['input_timesteps'] if 'input_timesteps' in layer else None
			input_dim = layer['input_dim'] if 'input_dim' in layer else None
			output_layer = layer['output_layer'] if 'output_layer' in layer else None

			if layer['type'] == 'dense':
				self.model.add(Dense(neurons, activation=activation))
			if layer['type'] == 'lstm':
				self.model.add(LSTM(neurons, stateful=True, batch_input_shape=(1,input_timesteps, input_dim), return_sequences=return_seq))
			if layer['type'] == 'dropout':
				self.model.add(Dropout(dropout_rate))
			if layer['type'] == 'TimeDistributed':
				self.model.add(TimeDistributed(Dense(output_layer)))

		self.model.summary()
		self.model.compile(loss=self.custom_loss, optimizer=configs['model_Pmimo']['optimizer'])

		print('[Model] Model Pmimo Compiled')
		timer.stop()

	def train(self, x, y, epochs, batch_size,verbose,shuffle):
		timer = Timer()
		timer.start()
		print('[Model] Training Started')
		print('[Model] %s epochs, %s batch size' % (epochs, batch_size))
		history= self.model.fit(x,
			y,
			epochs=epochs,
			batch_size=batch_size,
            verbose=verbose,
            shuffle=shuffle
		)
		timer.stop()
        
	def plotHystory (self, modelo):
		label = modelo
		plt.plot (self.history.history ["loss"] , label = "PMIMO")
		plt.ylabel('loss'); plt.xlabel('epoch')
		plt.grid()
		plt.legend ()
		plt.savefig("Comp_PMIMO.jpg")
		plt.show ()

	def predict (self, data):
		print('[Model] Predicting Point-by-Point...')
		predicted = self.model.predict(data)
		return predicted       
        
	def save (self):
		self.model.save('hiper.h5')
 
