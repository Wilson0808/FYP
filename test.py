# -*- coding: utf-8 -*-
"""
Created on Mon Dec 27 23:05:43 2021

@author: tanwi
"""

import numpy as np
import math


### (a) Generate info bits 
no_of_samples = 100000;
#no_of_samples = 1e2;
no_of_blocks = no_of_samples/4;
sequence = np.sign(np.random.randn(no_of_samples,1));
##### Convert +1 to "0" and -1 to "1" 
sequence_bits = (1-sequence)/2;
print(sequence_bits)
