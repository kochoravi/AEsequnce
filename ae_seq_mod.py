from keras.layers import Input, LSTM, RepeatVector, Lambda, Dense
from keras.models import Model
import keras.backend as K
from keras.optimizers import Adam
import os
# from keras.callbacks import ModelCheckpoint 
# from keras.utils import plot_model
import numpy as np
import tensorflow as tf

# input_dim = 4
# # timesteps = 3
# latent_dim = 2
# hidden_dim = 4

class Autoencoder():
    def __init__(self, input_dim, hidden_dim, latent_dim, use_dropout = False):
      self.name = 'autoencoder'
      self.input_dim = input_dim
      self.hidden_dim = hidden_dim
      self.latent_dim = latent_dim
      self.use_dropout = use_dropout
      self._build()

    def _build(self):
      def repeat(x):

        stepMatrix = K.ones_like(x[0][:,:,:1]) #matrix with ones, shaped as (batch, steps, 1)
        latentMatrix = K.expand_dims(x[1],axis=1) #latent vars, shaped as (batch, 1, latent_dim)

        return K.batch_dot(stepMatrix,latentMatrix)

      inputs = Input(shape=(None,input_dim), name = "input_layer")
      
      encoded_l1 = LSTM(hidden_dim, return_sequences=True, name = "elstm_I")(inputs)
      encoded_l2 = LSTM(latent_dim, name = "elstm_out")(encoded_l1)
      
      decoded_rep = Lambda(repeat, name = 'lambda')([inputs,encoded_l2])
      # decoded_rep = RepeatVector(timesteps)(encoded_l2)
      decoded_out = LSTM(input_dim, return_sequences=True, name = 'dlstm')(decoded_rep)
      
      #full Autoencoder
      self.autoencoder_model = Model(inputs, decoded_out, name = 'AE')
      #the encoder
      self.encoder_model = Model(inputs, encoded_l2, name = "encoder")
      
      # encoder = Model(inputs, encoded_l2, name = "encoder")
      # sequence_autoencoder.compile(optimizer='adam', loss ='mse')
      # encoder.summary()
      # sequence_autoencoder.summary()
    def train(self, sequence, batch_size, epochs, steps_per_epoch = 2, lr_decay = 1):
    
      def train_generator(sequence, batch = batch_size, window_size = 3):
        i = 0
        winbatch = batch*window_size
        while True:
            tempseq = np.array(sequence[i*winbatch: winbatch*(i+1),:])
            tempr = np.reshape(tempseq, (batch,window_size,-1))
            i+=1
        yield tempr, tempr
    
      self.autoencoder_model.fit(train_generator(x_train), steps_per_epoch = 2, epochs=5, verbose=1)
            
    def compile(self, learning_rate = 0.001 ):
      self.learning_rate = learning_rate
      optimizer = Adam(lr = learning_rate)
      self.autoencoder_model.compile(optimizer = optimizer, loss = 'mse')

      
      