# -*- coding: utf-8 -*-
"""
Created on Fri Apr  1 02:50:35 2022

@author: tanwi
"""

from operator import itemgetter
import sys
import random


def getEncoded(text):
    count = 0
    mp = {}
    for x in text:
        if x in mp:
            mp[x] += 1
        else:
            mp[x] = 1
        count += 1
    mpMain = []
    for key in mp:
        mpMain.append([key, mp[key], ''])
    # [['A', 5, ''], ['B', 4, ''], ...] example
    mpMain = sorted(mpMain, key=itemgetter(1), reverse=True)
    label(mpMain)
    mpMain = sorted(mpMain, key=itemgetter(0))
    # print(mpMain)
    encoded = ''
    dct = {}
    for x in mpMain:
        dct[x[0]] = x[2]
    for ch in text:
        encoded += dct[ch]

    return encoded, mpMain  # 'mpMain' is a list of each symbol with its encoded values, we need this list to decode
    # the encoded string correctly.


# 5 4 3 3 2 - 17
# 5 12 -- 7
# 9 8 -- 1

# sum allSum-sum
# abs(allSum - sum - sum )

def divide(mpDivide):
    allSum = 0
    for i in range(len(mpDivide)):
        allSum += mpDivide[i][1]
    i, mn, sum = 0, sys.maxsize, 0
    for k in range(len(mpDivide)):
        x = mpDivide[k]
        sum += x[1]
        if mn > abs(allSum - sum * 2):
            mn = abs(allSum - sum * 2)
            i = k + 1

    return mpDivide[:i], mpDivide[i:]


def label(mpLabel):  # to denote the right side with 1, left side with 0
    if len(mpLabel) > 1:
        mpL, mpR = divide(mpLabel)

        for i in range(len(mpL)):
            mpL[i][2] += '0'
        for i in range(len(mpR)):
            mpR[i][2] += '1'

        if len(mpL) == 1 and len(mpR) == 1:
            return
        label(mpL)
        label(mpR)


def binarySearch(mpSearch, target):
    s, e = 0, len(mpSearch) - 1
    while s <= e:
        m = s + (e - s) // 2
        if mpSearch[m][0] == target:
            return mpSearch[m][2]
        if target < mpSearch[m][0]:
            e = m - 1
        else:
            s = m + 1


def decode(encoded, mpMain):
    decodeChecker = {}
    for x in mpMain:
        decodeChecker[x[2]] = x[0]

    decoded = ''
    string = ''
    for x in encoded:
        string += x
        if decodeChecker.get(string):
            decoded += decodeChecker[string]
            string = ''
    return decoded


###############################
########HAMMING CODE###########
###############################


def divideDataBits(codes, encoded):
    bits = []
    for x in encoded:
        bits.append(x)
        if len(bits) == 4:
            codes.append(bits)
            bits = []
    while len(bits) < 4:
        bits.append('0')
    codes.append(bits)


# Hamming(7,4)
def HammingFunction(codes):
    hammingCode = ''
    hammingCodeWithSpaces = ''
    for bits in codes:
        r1 = int(bits[0]) ^ int(bits[1]) ^ int(bits[2])
        r2 = int(bits[1]) ^ int(bits[2]) ^ int(bits[3])
        r3 = int(bits[0]) ^ int(bits[1]) ^ int(bits[3])
        bits.extend([str(r1), str(r2), str(r3)])
        hammingCode += ''.join(map(str, bits))
        # hammingCodeWithSpaces += ''.join(map(str, bits)) + ' '

    # print(hammingCodeWithSpaces)
    return hammingCode


# To add random errors to our hamming code
def addErrors(hammingCode):
    block = ''
    newHammingCode = ''
    for x in hammingCode:
        block += x
        if len(block) == 7:
            n = random.randint(0, 7)
            if n == 7:
                newHammingCode += block
                block = ''
                continue
            else:
                helper = list(block)
                helper[n] = block[n] == '1' and '0' or '1'
                block = ''.join(helper)
                newHammingCode += block
                block = ''
    return newHammingCode


def getHammingCode(encoded):
    codes = []
    divideDataBits(codes, encoded)
    hammingCode = HammingFunction(codes)
    hammingCodeWithErrors = addErrors(hammingCode)
    return hammingCode, hammingCodeWithErrors  # This function returns a hamming code and a hamming code with random
    # errors at once


def getErrorBitIndex(s1, s2, s3):
    if not s1 and not s2 and not s3:
        return None
    elif not s1 and not s2 and s3:
        return 6
    elif not s1 and s2 and not s3:
        return 5
    elif not s1 and s2 and s3:
        return 3
    elif s1 and not s2 and not s3:
        return 4
    elif s1 and not s2 and s3:
        return 0
    elif s1 and s2 and not s3:
        return 2
    elif s1 and s2 and s3:
        return 1


# This function fixes any error in the Hamming code, if any.
def findAndFix(HCWithErrors):
    block = ''
    result = ''
    for x in HCWithErrors:
        block += x
        if len(block) == 7:
            S1 = int(block[4]) ^ int(block[0]) ^ int(block[1]) ^ int(block[2])
            S2 = int(block[5]) ^ int(block[1]) ^ int(block[2]) ^ int(block[3])
            S3 = int(block[6]) ^ int(block[0]) ^ int(block[1]) ^ int(block[3])
            i = getErrorBitIndex(S1, S2, S3)

            if i is not None:
                helper = list(block)
                helper[i] = block[i] == '1' and '0' or '1'
                block = ''.join(helper)

            result += block[0:4]
            block = ''

    return result


# driver code
def main():
    encoded, listWithCodeBlocks = getEncoded('Test input')
    print('encoded:', encoded)
    decoded = decode(encoded, listWithCodeBlocks)
    print('decoded:', decoded)

    # Hamming code (7,4)
    hammingCode, hammingCodeWithErrors = getHammingCode(encoded)
    print('hamming code: ', hammingCode)
    print('hamming code with errors', hammingCodeWithErrors)

    # Print results:
    # encoded: 01001100111001001011101110111100
    # decoded: Test input
    # hamming code:  010011111000101110100010011110110001011000101100011000100000000
    # hamming code with errors 000011111000111110110110011110010001001000101100011000100000000


if __name__ == '__main__':
    main()
