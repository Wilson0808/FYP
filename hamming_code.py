# -*- coding: utf-8 -*-
"""
Created on Mon Dec 27 18:52:20 2021

@author: tanwi
"""

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
no_of_samples = 40000;
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
HDdecode_info_BER = np.arange( 1.0,len(SNR_dB)+1)
HDdecode_info_BER = np.array(HDdecode_info_BER)
#print ('HDdecode_info_BER:')
#print(HDdecode_info_BER)
HDdecode_BLER = np.arange( 1.0,len(SNR_dB)+1)
HDdecode_BLER = np.array(HDdecode_BLER)
#print ('HDdecode_BLER:')
#print(HDdecode_BLER)
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

for counter in range( 0,len(SNR_dB)):
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
    Rx_block = (1-np.sign(Rx_block_1))/2;
    #print ('Rx_block:')
    #print(Rx_block)

    # Raw code-bit error rate
    Rx_block = np.array(Rx_block, dtype = bool)
    #print ('Rx_block:')
    #print(Rx_block)
    code_block = np.array(code_block, dtype = bool)
    #print ('code_block:')
    #print(code_block)
    errors = Rx_block!=code_block;
    #print ('errors:')
    #print(errors)
    Rx_block = np.array(Rx_block, dtype = int)
    #print ('Rx_block:')
    #print(Rx_block)
    code_block = np.array(code_block, dtype = int)
    #print ('code_block:')
    #print(code_block)
    errors = np.array(errors, dtype = int)
    #print ('errors:')
    #print(errors)
    no_of_codebit_errors = np.sum(errors)
    #print ('no_of_codebit_errors:')
    #print(no_of_codebit_errors)
    raw_codebit_BER[counter] = no_of_codebit_errors/(no_of_samples/rate);
    #print ('raw_codebit_BER[counter]:')
    #print(raw_codebit_BER[counter])

    # Raw code-block error rate
    #no_of_errors_eachblock = sum(errors,2);
    for eeee in range( 0,int(no_of_blocks)):
        no_of_errors_eachblock[eeee] = sum(errors[eeee])
    #print ('no_of_errors_eachblock:')
    #print(no_of_errors_eachblock)
    no_of_codeblock_errors = sum(np.sign(no_of_errors_eachblock))
    #print ('no_of_codeblock_errors:')
    #print(no_of_codeblock_errors)
    raw_BLER[counter] = no_of_codeblock_errors/int(no_of_blocks)
    #print ('raw_BLER[counter]:')
    #print(raw_BLER[counter])

    ### perform ML hard decoding on Rx_block_1 
    Correlation_value = np.dot(np.sign(Rx_block_1),code_bit_1.T);
    #print ('Correlation_value:')
    #print(Correlation_value)
    
    for block in range(0,int(no_of_blocks)):
        #print ('block:')
        #print(block)
        M[block] = max(Correlation_value[block]);
        for xxxx in range(0,16):    
            #print (xxxx)
            #print ('Correlation_value[block,xxxx]:')
            #print(Correlation_value[block,xxxx])
            if (Correlation_value[block,xxxx] == M[block]):
                I[block] = xxxx
                #print ('xxxx:')
                #print(xxxx)
            #else: I[block] = 0
        #print ('M[block]:')
        #print(M[block])
        #print ('I[block]:')
        #print(I[block])
        HDdecode_code_block[block] = code_bit[I[block]];
        #print ('HDdecode_code_block[block]:')
        #print(HDdecode_code_block[block])
    #print ('HDdecode_code_block:')
    #print(HDdecode_code_block)
    
    # extract info blocks after ML hard decoding
    HDdecode_info_block = [list(zip(*HDdecode_code_block))[0], list(zip(*HDdecode_code_block))[1], list(zip(*HDdecode_code_block))[2], list(zip(*HDdecode_code_block))[4]];
    HDdecode_info_block = np.array(HDdecode_info_block).T
    #print ('HDdecode_info_block:')
    #print(HDdecode_info_block)

    # Decoded code-bit error rate
    HDdecode_errors = HDdecode_code_block != code_block;
    HDdecode_errors = np.array(HDdecode_errors, dtype = int)
    #print ('HDdecode_errors:')
    #print(HDdecode_errors)
    no_of_HDdecode_codebit_errors = np.sum(HDdecode_errors)
    #print ('no_of_HDdecode_codebit_errors:')
    #print(no_of_HDdecode_codebit_errors)
    HDdecode_codebit_BER[counter] = no_of_HDdecode_codebit_errors/(no_of_samples/rate);
    #print ('HDdecode_codebit_BER[counter]:')
    #print(HDdecode_codebit_BER[counter])

    # Decoded info-bit error rate
    HDdecode_info_errors = HDdecode_info_block != info_block;
    HDdecode_info_errors = np.array(HDdecode_info_errors, dtype = int)
    #print ('HDdecode_info_errors:')
    #print(HDdecode_info_errors)
    no_of_HDdecode_info_errors = np.sum(HDdecode_info_errors);
    #print ('no_of_HDdecode_info_errors:')
    #print(no_of_HDdecode_info_errors)
    HDdecode_info_BER[counter] = no_of_HDdecode_info_errors/no_of_samples;
    #print ('HDdecode_info_BER[counter]:')
    #print(HDdecode_info_BER[counter])

    # Decoded code-block error rate
    #no_of_HDdecode_errors_eachblock = sum(HDdecode_errors,2);
    for nohee in range( 0,int(no_of_blocks)):
        no_of_HDdecode_errors_eachblock[nohee] = sum(HDdecode_errors[nohee])
    #print ('no_of_HDdecode_errors_eachblock:')
    #print(no_of_HDdecode_errors_eachblock)
    no_of_HDdecode_codeblock_errors = sum(np.sign(no_of_HDdecode_errors_eachblock));
    #print ('no_of_HDdecode_codeblock_errors:')
    #print(no_of_HDdecode_codeblock_errors)
    HDdecode_BLER[counter] = no_of_HDdecode_codeblock_errors/(no_of_blocks);
    #print ('HDdecode_BLER[counter]:')
    #print(HDdecode_BLER[counter])

#print ('raw_codebit_BER, raw_BLER, HDdecode_codebit_BER, HDdecode_info_BER, HDdecode_BLER:')
#print(raw_codebit_BER, raw_BLER, HDdecode_codebit_BER, HDdecode_info_BER, HDdecode_BLER)
#figure
plot.grid(True, which ="both")
#plot.semilogy(SNR_dB, raw_codebit_BER, SNR_dB, raw_BLER, SNR_dB, HDdecode_codebit_BER, SNR_dB, HDdecode_info_BER, SNR_dB, HDdecode_BLER);
#plot.legend({'Raw code-bit error rate', 'Raw block error rate', 'ML hard decoded code-bit error rate', 'ML hard decoded information-bit error rate', 'ML hard decoded block error rate'}, loc='lower left')
plot.semilogy(SNR_dB, raw_codebit_BER, marker = '^', linewidth = "0.25", linestyle = "dashed", label = 'Raw code-bit error rate');
plot.semilogy(SNR_dB, raw_BLER, marker = 'o', linewidth = "0.5", label = 'Raw block error rate');
plot.semilogy(SNR_dB, HDdecode_codebit_BER, marker = '^', linewidth = "0.25", linestyle = "dashed", label = 'ML hard decoded code-bit error rate');
plot.semilogy(SNR_dB, HDdecode_info_BER, marker = '.', linewidth = "0.75", label = 'ML hard decoded information-bit error rate');
plot.semilogy(SNR_dB, HDdecode_BLER, marker = 'o', linewidth = "0.5", label = 'ML hard decoded block error rate');
plot.legend(fontsize='7', loc='lower left')
#plot.ylim(0,1)
plot.xlim(0,5)
#plot.semilogy(SNR_dB, raw_codebit_BER, 'b*', SNR_dB, raw_BLER, 'bO-', SNR_dB, HDdecode_codebit_BER, 'm*', SNR_dB, HDdecode_info_BER, 'm-.', SNR_dB, HDdecode_BLER, 'mO-');
#plot.legend({'Raw code-bit error rate', 'Raw block error rate', 'ML hard decoded code-bit error rate', 'ML hard decoded information-bit error rate', 'ML hard decoded block error rate'},'Location','SouthWest');
#plot.axis([0,5,0,1])
plot.xlabel('SNR in dB for AWGN channel');
plot.ylabel('Simulated Error Rate');
plot.title('ML decoding of a (7,4) Hamming coded system over AWGN channel');
plot.show()
#grid

#print('HammingCoded_MLHD_AWGN.eps','-depsc')

toc = time.time()
tElapsed = toc - tic
#print ('tElapsed:')
#print(tElapsed)

