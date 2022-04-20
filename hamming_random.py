# -*- coding: utf-8 -*-
"""
Created on Sat Dec 18 16:31:01 2021

@author: tanwi
"""

# Hamming code(7,4)

import random

# Generator Matrix
# G = [1, 0, 0, 1, 0, 1, 1;
#      0, 1, 0, 1, 0, 1, 0;
#      0, 0, 1, 1, 0, 0, 1;
#      0, 0, 0, 0, 1, 1, 1]
# Encoding matrix
G = ['1101', '1011', '1000', '0111', '0100', '0010', '0001']

# Parity Check Matrix
# H = [1, 1, 1, 1, 0, 0, 0;
#      1, 1, 0, 0, 1, 1, 0;
#      1, 0, 1, 0, 1, 0, 1]
H = ['1010101', '0110011', '0001111']

# Bit position 
# B = [1, 1, 1;
#      1, 1, 0;
#      1, 0, 1;
#      1, 0, 0;
#      0, 1, 1;
#      0, 1, 0;
#      0, 0, 1]
B = ['100', '010', '110', '001', '101', '011', '111']

# Error pattern
# E = [0, 0, 0, 0, 0, 0, 1;
#      0, 0, 0, 0, 0, 1, 0;
#      0, 0, 0, 0, 1, 0, 0;
#      0, 0, 0, 1, 0, 0, 0;
#      0, 0, 1, 0, 0, 0, 0;
#      0, 1, 0, 0, 0, 0, 0;
#      1, 0, 0, 0, 0, 0, 0]
# Decoding matrix
E = ['0010000', '0000100', '0000010', '0000001']

# Input 4-bit string
p = ''.join([random.choice('01') for k in range(4)])
print ('Input bit string: ' + p)

#Encoding bit string by Encoding matrix
x = ''.join([str(bin(int(i, 2) & int(p, 2)).count('1') % 2) for i in G])
print ('Encoded bit string to send: ' + x)

# add 1 bit error
e = random.randint(0, 7)

# Counted from left starting from 1, find bit error
print ('Which bit got error during transmission (0: no error): ' + str(e))
if e > 0:
    x = list(x)
    x[e - 1] = str(1 - int(x[e - 1]))
    x = ''.join(x)
print ('Encoded bit string that got error during tranmission: ' + x )

# Find bit error 
z = ''.join([str(bin(int(j, 2) & int(x, 2)).count('1') % 2) for j in H])
if int(z, 2) > 0:
    e = int(B[int(z, 2) - 1], 2)
else:
    e = 0
print ('Which bit found to have error (0: no error): ' + str(e))

# correct the error
if e > 0:
    x = list(x)
    x[e - 1] = str(1 - int(x[e - 1]))
    x = ''.join(x)

p = ''.join([str(bin(int(k, 2) & int(x, 2)).count('1') % 2) for k in E])
print ('Corrected output bit string: ' + p)