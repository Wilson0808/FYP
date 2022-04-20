# -*- coding: utf-8 -*-
"""
Created on Mon Apr 11 10:11:07 2022

@author: tanwi
"""
from tensorflow.keras import models, layers, utils, backend as K
import matplotlib.pyplot as plt

# define the function
import tensorflow as tf

#def binary_step_activation(x):
#    ##return 1 if x>0 else 0 
#    return K.switch(x>0, tf.math.divide(x,x), tf.math.multiply(x,0))




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
no_of_samples = 4000;
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

for counter in range( 0,len(SNR_dB)-5):
    print ('counter:')
    print(counter)
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




#n_features = 7
## DeepNN
#### layer input
#inputs = layers.Input(name="input", shape=(n_features,))
#### hidden layer 1
#h1 = layers.Dense(name="h1", units=int(round((n_features))), activation='relu')(inputs)
#h1 = layers.Dropout(name="drop1", rate=0.2)(h1)
#### hidden layer 2
#h2 = layers.Dense(name="h2", units=int(round((n_features))), activation='relu')(h1)
#h2 = layers.Dropout(name="drop2", rate=0.2)(h2)
### layer output
#outputs = layers.Dense(name="output", units=7, activation='linear')(h2)
#model = models.Model(inputs=inputs, outputs=outputs, name="DeepNN")
#model.summary()


n_features = 7
model = models.Sequential(name="DeepNN", layers=[
    ### hidden layer 1
    layers.Dense(name="h1", input_dim=n_features,
                 units=int(round((n_features))), 
                 activation='relu'),
    layers.Dropout(name="drop1", rate=0.2),
    
    ### hidden layer 2
    layers.Dense(name="h2", 
                 units=int(round((n_features))), 
                 activation='relu'),
    layers.Dropout(name="drop2", rate=0.2),
    
    ### layer output
    layers.Dense(name="output", units=7, activation='linear')
])
model.summary()




# define metrics
def Recall(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def Precision(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def F1(y_true, y_pred):
    precision = Precision(y_true, y_pred)
    recall = Recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

# compile the neural network
model.compile(optimizer='adam', loss='binary_crossentropy', 
              metrics=['accuracy',F1])




# define metrics
def R2(y, y_hat):
    ss_res =  K.sum(K.square(y - y_hat)) 
    ss_tot = K.sum(K.square(y - K.mean(y))) 
    return ( 1 - ss_res/(ss_tot + K.epsilon()) )

# compile the neural network
model.compile(optimizer='adam', loss='mean_absolute_error', 
              metrics=[R2])




import numpy as np
X = Rx_block_1
y = code_block




# train/validation
training = model.fit(x=X, y=y, batch_size=32, epochs=100, shuffle=True, verbose=0, validation_split=0.3)

# plot
metrics = [k for k in training.history.keys() if ("loss" not in k) and ("val" not in k)]    
fig, ax = plt.subplots(nrows=1, ncols=2, sharey=True, figsize=(15,3))
       
## training    
ax[0].set(title="Training")    
ax11 = ax[0].twinx()    
ax[0].plot(training.history['loss'], color='black')    
ax[0].set_xlabel('Epochs')    
ax[0].set_ylabel('Loss', color='black')    
for metric in metrics:        
    ax11.plot(training.history[metric], label=metric)    
    ax11.set_ylabel("Score", color='steelblue')    
ax11.legend()
        
## validation    
ax[1].set(title="Validation")    
ax22 = ax[1].twinx()    
ax[1].plot(training.history['val_loss'], color='black')    
ax[1].set_xlabel('Epochs')    
ax[1].set_ylabel('Loss', color='black')    
for metric in metrics:          
    ax22.plot(training.history['val_'+metric], label=metric)    
    ax22.set_ylabel("Score", color="steelblue")    
plt.show()

#probability_model = tf.keras.Sequential([model,tf.keras.layers.Softmax()])
predictions = model.predict(Rx_block_1)
#print ('predictions:')
#print(predictions)


