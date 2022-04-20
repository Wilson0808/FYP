# -*- coding: utf-8 -*-
"""
Created on Fri Apr  1 02:44:11 2022

@author: tanwi
"""

import numpy as np
import re

G = np.array([[1, 1, 0, 1], [1, 0, 1, 1], [1, 0, 0, 0], [0, 1, 1, 1], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
H = np.array([[1, 0, 1, 0, 1, 0, 1], [0, 1, 1, 0, 0, 1, 1], [0, 0, 0, 1, 1, 1, 1]])


def code(code_b):
    code_b = list(map(int, str(code_b)))
    Gb = np.dot(G, code_b, out=None)
    code_k = np.mod(Gb, 2)
    print('Gb:  ', Gb, '\nk:    ', code_k)
    return


def decode(decode_k):
    decode_k = list(map(int, str(decode_k)))
    Hk = np.dot(H, decode_k, out=None)
    e = np.mod(Hk, 2)
    e_string = ''.join(str(i) for i in e.tolist())

    if e_string == '000':
        print('Hk:  ', Hk, '\ne:    ', e)
    else:
        place = int(e_string, 2)
        corrected_code = decode_k
        wrong_bit = decode_k[place-1]

        if wrong_bit == 1:
            corrected_code[place-1] = 0

        if wrong_bit == 0:
            corrected_code[place - 1] = 1

        corrected_code = ''.join(str(i) for i in corrected_code)
        print('error detected in place ', place, '\ncorrected code is: ', corrected_code)

    return


def invalid_input():
    print('Please enter a valid input')


if __name__ == '__main__':
    res = ''
    while res != 'q':
        res = input('\nCode(1) or Decode(2)? (q to quit): ').strip()

        if res == 'q':
            exit()

        if res == '1':
            b = input('input your data, 4 bits: ').strip()
            if b.isnumeric() and len(b) == 4 and re.match('^[0-1]+$', b):
                code(b)
            else:
                invalid_input()

        elif res == '2':
            k = input('input your data, 7 bits: ').strip()

            if k.isnumeric() and len(k) == 7 and re.match('^[0-1]+$', k):
                decode(k)
            else:
                invalid_input()

        else:
            invalid_input()