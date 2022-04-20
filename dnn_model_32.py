# -*- coding: utf-8 -*-
"""
Created on Mon Apr 11 10:11:07 2022

@author: tanwi
"""
from tensorflow.keras import models, layers, utils, backend as K
import matplotlib.pyplot as plt
from tensorflow import keras
import tensorflow as tf
import numpy as np
import math
import time
import matplotlib.pyplot as plot

##############Hamming code(7,4)############
######Generator Matrix######
# G = [1, 0, 0, 1, 0, 1, 1;
# 0, 1, 0, 1, 0, 1, 0;
# 0, 0, 1, 1, 0, 0, 1;
# 0, 0, 0, 0, 1, 1, 1];
######Parity Check Matrix######
# H = [1, 1, 1, 1, 0, 0, 0;
# 1, 1, 0, 0, 1, 1, 0;
# 1, 0, 1, 0, 1, 0, 1];

#close all
#clear all
tic = time.time(); 

# Generator Matrix
G = np.array([[1, 0, 0, 1, 0, 1, 1],[0, 1, 0, 1, 0, 1, 0],[0, 0, 1, 1, 0, 0, 1],[0, 0, 0, 0, 1, 1, 1]])
#print ('G:')
#print(G)
#####Parity Check Matrix######
H = np.array([[1, 1, 1, 1, 0, 0, 0], [1, 1, 0, 0, 1, 1, 0], [1, 0, 1, 0, 1, 0, 1]])
#print ('H:')
#print(H)
#code rate
rate = 4/7;
 
### Generate all possible input sequences 
k=4;
#print ('k:')
#print(k)
data_bit = np.zeros((pow(2,k),k));
#print ('data_bit:')
#print(data_bit)
data = list(range(0,pow(2,k)))
#print ('data:')
#print(data)
for d in range(0,pow(2,k)):
    data[d] = int(format(d,"b"))
    #print ('data[d]:')
    #print(data[d])
data_temp = np.array(data);
#print ('data_temp')
#print(data_temp)
data_temp = data;
#print ('data_temp')
#print(data_temp)
for r in range(0,k):
    for c in range(0,pow(2,k)):
        data_bit[c,r] = math.floor(data[c]/pow(10,(k-1-r)))%10
        #print ('data_bit[c,r]')
        #print(data_bit[c,r])
        data_temp[c] = data_temp[c]%pow(10,(k-r));
        #print ('data_temp[c]')
        #print(data_temp[c])
#print ('data_bit')
#print(data_bit)        
### Generate all possible output sequences 
code_bit = ((np.dot(data_bit,G))%2);
#print ('code_bit')
#print(code_bit)
### Generate all possible output sequences in terms of +1 ("0") and -1 ("1")
code_bit_1 = pow(-1,code_bit);
#print ('code_bit_1')
#print(code_bit_1)
#############################
### (a) Generate info bits 
no_of_samples = 12800;
#print ('no_of_samples')
#print(no_of_samples)
#no_of_samples = 100;
no_of_blocks = no_of_samples/k;
#print ('no_of_blocks')
#print(no_of_blocks)
sequence = np.sign(np.random.randn(no_of_samples,1));
#print ('sequence')
#print(sequence)
##### Convert +1 to "0" and -1 to "1" 
sequence_bits = (1-sequence)/2;
#print ('sequence_bits')
#print(sequence_bits)
### (b) apply Hamming encoding
#info_block = np.reshape(sequence_bits,[no_of_blocks,k]);
info_block = sequence_bits.reshape((int(no_of_blocks),k));
#print ('info_block:')
#print(info_block)
code_block = ((np.dot(info_block,G))%2);
#print ('code_block:')
#print(code_block)
code_block_1 = pow(-1,code_block);
#print ('code_block_1:')
#print(code_block_1)

SNR_dB = list(range(0,6))
#print ('SNR_dB:')
#print(SNR_dB)
SNR = list(range(0,6))
#print ('SNR:')
#print(SNR)
noise_power = list(range(0,6))
#print ('noise_power:')
#print(noise_power)
for s in range(0,6):
    SNR[s] = pow(10,(int(s)/10))
    #print ('SNR[s]:')
    #print(SNR[s])
    noise_power[s] = 1./SNR[s]; 
    #print ('noise_power[s]:')
    #print(noise_power[s])
noise_power = np.array(noise_power)
#print ('SNR:')
#print(SNR)
#print ('noise_power:')
#print(noise_power)
#SNR = pow(10,(SNR_dB/10));
#noise_power = 1./SNR;
n = np.random.randn(*np.shape(code_block_1));
#print ('n:')
#print(n)
#counter
raw_codebit_BER = np.arange( 1.0,len(SNR_dB)+1)
raw_codebit_BER = np.array(raw_codebit_BER)
#print ('raw_codebit_BER:')
#print(raw_codebit_BER)
raw_BLER = np.arange( 1.0,len(SNR_dB)+1)
raw_BLER = np.array(raw_BLER)
#print ('raw_BLER:')
#print(raw_BLER)
HDdecode_codebit_BER = np.arange( 1.0,len(SNR_dB)+1)
HDdecode_codebit_BER = np.array(HDdecode_codebit_BER)
#print ('HDdecode_codebit_BER:')
#print(HDdecode_codebit_BER)
STdecode_codebit_BER = np.arange( 1.0,len(SNR_dB)+1)
STdecode_codebit_BER = np.array(STdecode_codebit_BER)
#print ('STdecode_codebit_BER:')
#print(STdecode_codebit_BER)
HDdecode_info_BER = np.arange( 1.0,len(SNR_dB)+1)
HDdecode_info_BER = np.array(HDdecode_info_BER)
#print ('HDdecode_info_BER:')
#print(HDdecode_info_BER)
STdecode_info_BER = np.arange( 1.0,len(SNR_dB)+1)
STdecode_info_BER = np.array(STdecode_info_BER)
#print ('STdecode_info_BER:')
#print(STdecode_info_BER)
HDdecode_BLER = np.arange( 1.0,len(SNR_dB)+1)
HDdecode_BLER = np.array(HDdecode_BLER)
#print ('HDdecode_BLER:')
#print(HDdecode_BLER)
STdecode_BLER = np.arange( 1.0,len(SNR_dB)+1)
STdecode_BLER = np.array(STdecode_BLER)
#print ('STdecode_BLER:')
#print(STdecode_BLER)
#block
M = range(1,int(no_of_blocks)+1)
M = np.array(M)
#print ('M:')
#print(M)
I = range(1,int(no_of_blocks)+1)
I = np.array(I)
#print ('I:')
#print(I)
no_of_errors_eachblock = range(1,int(no_of_blocks)+1)
no_of_errors_eachblock = np.array(no_of_errors_eachblock)
no_of_HDdecode_errors_eachblock = range(1,int(no_of_blocks)+1)
no_of_HDdecode_errors_eachblock = np.array(no_of_HDdecode_errors_eachblock)
for yyyy in range( 0,int(no_of_blocks)):
    M[yyyy]=100
    I[yyyy]=100
    no_of_errors_eachblock[yyyy]=0
    no_of_HDdecode_errors_eachblock[yyyy]=0
HDdecode_code_block = np.zeros((int(no_of_blocks),7));
HDdecode_code_block = np.array(HDdecode_code_block)
#print ('HDdecode_code_block:')
#print(HDdecode_code_block)

for counter in range( 0,len(SNR_dB)-3):
    #print ('counter:')
    #print(counter)
    noise = math.sqrt(noise_power[counter])*n;
    #print ('noise:')
    #print(noise)
    noise = noise.reshape((int(no_of_blocks)),7);
    #print ('noise:')
    #print(noise)
    # received code blocks
    Rx_block_1 = code_block_1+noise;
    #print ('Rx_block_1:')
    #print(Rx_block_1)



def get_model():
    # Create a simple model.
    n_features = 7
    inputs = layers.Input(name="input", shape=(n_features,))
    h1 = layers.Dense(name="h1", units=int(128), activation='relu')(inputs)
    h2 = layers.Dense(name="h2", units=int(128), activation='relu')(h1)
    outputs = layers.Dense(name="output", units=7, activation='linear')(h2)
    model = models.Model(inputs=inputs, outputs=outputs, name="DeepNN")
    model.compile(optimizer="adam", loss="mean_squared_error")
    return model


model = get_model()
model.summary()
# Train the model.
for block in range(0,int(no_of_blocks)):
    if (block == no_of_blocks/32):
        break
    test_input = Rx_block_1[block*32:block*32+32]
    test_target = code_block[block*32:block*32+32]
    training = model.fit(test_input, test_target, batch_size=32, epochs=100)
    

# Calling `save('my_model')` creates a SavedModel folder `my_model`.
model.save("my_model12800")

# It can be used to reconstruct the model identically.
#reconstructed_model = keras.models.load_model("my_model")

# Let's check:
#np.testing.assert_allclose(
#    model.predict(test_input), reconstructed_model.predict(test_input)
#)

# The reconstructed model is already compiled and has retained the optimizer
# state, so training can resume:
#reconstructed_model.fit(test_input, test_target)

#plot graph
#acc = training.history['acc']
#val_acc = training.history['val_acc']
#loss = training.history['loss']
#val_loss = training.history['val_loss']

#epochs = range(0, 100)
#plt.plot(epochs, loss , color='black', label='Training loss')
#plt.plot(epochs, acc, color='steelblue', label='Training acc')
#plt.title('Training')
#plt.legend()
#plt.figure()
#plt.plot(epochs, val_loss , color='black', label='Validation loss')
#plt.plot(epochs, val_acc, color='steelblue', label='Validation acc')
#plt.title('Validation')
#plt.legend()
#plt.show()



