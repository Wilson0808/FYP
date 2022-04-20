# -*- coding: utf-8 -*-
"""
Created on Fri Apr  1 02:58:17 2022

@author: tanwi
"""

import math


def int_to_bit(x):
    arr = []
    for i in range(16):
        arr.append(x >> i & 1)
    arr.reverse()
    return arr


def bin_to_int(list_bits):
    x = 0
    for i in range(len(list_bits)):
        x += list_bits[i] * 2 ** (len(list_bits) - i - 1)
    return x


def check_if_even(num):
    number_of_repeat = int(math.sqrt(len(int_to_bit(num))))
    list_of_bites = int_to_bit(num)
    for i in range(number_of_repeat):
        size_of_returns_list = int(len(list_of_bites) / 2)
        list_of_bites = divide_and_check([], list_of_bites, size_of_returns_list, True)
    return list_of_bites[0]


def divide_and_check(returns_list, list_bit, size_of_returns_list, is_stop):
    left_part = []
    right_part = []
    for i in range(int(len(list_bit) / 2)):
        left_part.append(list_bit[i])
        right_part.append(list_bit[int(len(list_bit) / 2) + i])
    if int(len(left_part)) != 1:
        divide_and_check(returns_list, left_part, size_of_returns_list, False)
    if int(len(right_part)) != 1:
        divide_and_check(returns_list, right_part, size_of_returns_list, False)
    if int(len(right_part)) == 1 and int(len(left_part)) == 1:
        returns_list.append(check(left_part[0], right_part[0]))
    if int(len(returns_list)) == size_of_returns_list and is_stop:
        return returns_list


def check(x, y):
    return 0 if x == y else 1


def trim_message(message):
    new_message = []
    z = 0
    for i in range(int(len(message))):
        if message[i] == 1:
            z = i
            break
    for i in range(z, int(len(message)), 1):
        new_message.append(message[i])
    return new_message


def calculate_z(message, x):
    check_list = []
    for z in range(2 ** x):
        for i in range(z + 2 ** x - 1, int(len(message)), (2 ** x) * 2):
            if i == 0 or i == 1 or i == 3 or i == 7:
                continue
            check_list.append(message[i])
    return check_if_even(bin_to_int(check_list))


def get_hamming_code(message):
    list_of_hamming_code = [0 for i in range(int(len(message)) + 4)]
    index = 0
    for i in range(int(len(list_of_hamming_code))):
        if i != 0 and i != 1 and i != 3 and i != 7:
            list_of_hamming_code[i] = message[index]
            index += 1
        else:
            list_of_hamming_code[i] = 0

    list_of_hamming_code[0] = calculate_z(list_of_hamming_code, 0)
    list_of_hamming_code[1] = calculate_z(list_of_hamming_code, 1)
    list_of_hamming_code[3] = calculate_z(list_of_hamming_code, 2)
    list_of_hamming_code[7] = calculate_z(list_of_hamming_code, 3)

    return list_of_hamming_code


def get_message_from_hamming_code(hamming_code):
    message = [0 for i in range(int(len(hamming_code)) - 4)]
    index = 0
    for i in range(int(len(hamming_code))):
        if i != 0 and i != 1 and i != 3 and i != 7:
            message[index] = hamming_code[i]
            index += 1
    return message


def is_message_broken(hamming_code):
    syndrome = [0 for i in range(4)]
    if calculate_z(hamming_code, 0) != hamming_code[0]:
        syndrome[3] = 1
    if calculate_z(hamming_code, 1) != hamming_code[1]:
        syndrome[2] = 1
    if calculate_z(hamming_code, 2) != hamming_code[3]:
        syndrome[1] = 1
    if calculate_z(hamming_code, 3) != hamming_code[7]:
        syndrome[0] = 1
    if syndrome == [0, 0, 0, 0]:
        return False
    else:
        print("syndrome == " + str(syndrome))
        return True


number = 15
print("number == " + str(number))
print("message before coding == " + str(trim_message(int_to_bit(number))))
print("message after coding == " + str(get_hamming_code(trim_message(int_to_bit(number)))))
print("message after decoding == " + str(
    get_message_from_hamming_code(get_hamming_code(trim_message(int_to_bit(number))))))
print("is message broken == " + str(is_message_broken(get_hamming_code(trim_message(int_to_bit(number))))))
print()

broken_message = [1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0]
print("broken message after coding == " + str(broken_message))
print("broken message after decoding == " + str(
    get_message_from_hamming_code(broken_message)))
print("is message broken == " + str(is_message_broken(broken_message)))