import random
import re
from math import log

alpha_list = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 
             'p','q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', ' ']


# char_to_num, takes a piece of text (a string of lowercase letters) and returns a list of the numerical values for each number, based on the indices in alpha_list. 
def char_to_num(char_text):
    num_text = []
    for letter in char_text:
        num_text.append(alpha_list.index(letter))
    return num_text

# num_to_char, reverses the effect, going from a series of numbers (ranged 0 - 26) to a string of lowercase letters returned as a list.
def num_to_char(num_text):
    char_text = []
    for num in num_text:
        char_text.append(alpha_list[num])
    return char_text


def encode_caesar_cipher(plaintext, key):
    ciphertext_num = []
    plaintext_num = char_to_num(plaintext)
    for num in plaintext_num:
        if num != 26: #num 26 is a space
            num += key
            num = num % 26
        ciphertext_num.append(num)
    ciphertext = num_to_char(ciphertext_num)
    return ciphertext

def decode_caesar_cipher(ciphertext, key):
    plaintext_num = []
    ciphertext_num = char_to_num(ciphertext)
    for num in ciphertext_num:
        if num != 26: #num 26 is a space
            num -= key
            num = num % 26
        plaintext_num.append(num)
    plaintext = num_to_char(plaintext_num)
    return plaintext


# Aristocrat


# returns a key, a list of numbers, random permutation of 0-25
def make_rand_aris_key():
    nums = []
    for i in range(26):
        nums.append(i)
    num_key = []
    for i in range(26):
        num = (random.choice(nums))
        num_key.append(num)
        nums.remove(num)
    return num_key

def decode_aris(ciphertext, num_key):
    plaintext_num = []
    ciphertext_num = char_to_num(ciphertext)
    for num in ciphertext_num:
        if num != 26:
            num = num_key.index(num)
        plaintext_num.append(num)
    plaintext = num_to_char(plaintext_num)
    return plaintext

# plaintext: string or list of characters
# num_key is a list of numbers 
# returns a encoded list of characters
def encode_aris(plaintext, num_key):
    ciphertext_num = []
    plaintext_num = char_to_num(plaintext)
    for num in plaintext_num:
        if num != 26:
            num = num_key[num]
        ciphertext_num.append(num)
    ciphertext = num_to_char(ciphertext_num)
    return ciphertext


def count_letters(text):
    letter_count = []
    for i in range(27):
        letter_count.append(0)
    for letter in text:
        letter_count[alpha_list.index(letter)] += 1
    return letter_count

def count_letter_pairs(text):
    pair_count = []
    for i in range(27):
        pair_count.append([])
        for l2 in range(27):
            pair_count[i].append(0)
    for i in range(len(text) - 1):
        pair_count[alpha_list.index(text[i])][alpha_list.index(text[i + 1])] += 1
    return pair_count


def normalize_counts(count_list, pseudo_count = 0):
    total = 0
    normalized_counts = []
    for i in range(27):
        total += (count_list[i] + pseudo_count)
        normalized_counts.append(0)
    for i in range(27):
        normalized_counts[i] = (count_list[i] + pseudo_count) / total
    return normalized_counts

def normalize_counts_no_spaces(count_list):
    total = 0
    normalized_counts = []
    for i in range(26):
        total += count_list[i]
        normalized_counts.append(0)
    for i in range(26):
        normalized_counts[i] = count_list[i] / total
    return normalized_counts

def compute_transition_matrix(pair_count, pseudo_count = 0):
    transition_matrix = []
    for i in range(27):
        transition_matrix.append([])
    for i in range(27):
        transition_matrix[i] = (normalize_counts(pair_count[i], pseudo_count))
    return transition_matrix

def make_letters(file_name):
    new_file = open(file_name, 'r')
    file = new_file.read()
    file = file.lower()
    #regular expressions is imported
    #The next line takes out only the letters
    return re.findall('[a-z]|\ ', file)





def find_likelihood(text, percent_list):
    likely = 1
    for letter in text:
        letter = alpha_list.index(letter)
        likely *= percent_list[letter]
    return likely

def find_log_likelihood(text, percent_list):
    log_likely = 0
    for letter in text:
        letter = alpha_list.index(letter)
        log_likely += log(percent_list[letter])
    return log_likely

def find_pair_log_likelihood(text, ref_percent_list, ref_transition_matrix):
    log_likely = 0
    log_likely += log(ref_percent_list[alpha_list.index(text[0])])
    for letter in range(len(text) - 1):
        first_index = alpha_list.index(text[letter])
        second_index = alpha_list.index(text[letter + 1])
        if ref_transition_matrix[first_index][second_index] != 0:
            log_likely += log(ref_transition_matrix[first_index][second_index])
    return log_likely

    
def calculate_log_likelihood_change(text, text_pair_count, ref_letter_percent,\
                                ref_transition_matrix, index1, index2, key):
    column_change, row_change = 0, 0
    l1 = int(key[index1])
    l2 = int(key[index2])
    for j in range(27):
        if j != l1 and j != l2:
            column_change += (text_pair_count[l1][j] - text_pair_count[l2][j])\
            * (log(ref_transition_matrix[l1][j]) - log(ref_transition_matrix[l2][j]))
    for i in range(27):
        if i != l1 and i != l2:
            row_change += (text_pair_count[i][l1] - text_pair_count[i][l2])\
            * (log(ref_transition_matrix[i][l1]) - log(ref_transition_matrix[i][l2]))
    
    change = column_change + row_change
    
    change += (text_pair_count[l1][l1] - text_pair_count[l2][l2])\
    * (log(ref_transition_matrix[l1][l1]) - log(ref_transition_matrix[l2][l2]))
    
    change += (text_pair_count[l1][l2] - text_pair_count[l2][l1])\
    * (log(ref_transition_matrix[l1][l2]) - log(ref_transition_matrix[l2][l1]))
    #dealing with the first letter (special case)
    
    if text[0] == alpha_list[l1]:
        change += log(ref_letter_percent[l1]) - log(ref_letter_percent[l2])
    elif text[0] == alpha_list[l2]:
        change += log(ref_letter_percent[l2]) - log(ref_letter_percent[l1])
    return -1 * change
    
    
    
def switch_letters(letters, l1, l2):
    text = list(letters)
    for i in range(len(text)):
        if text[i] == l1:
            text[i] = l2
        elif text[i] == l2:
            text[i] = l1
    return text

def switch_row_and_columns(text_pair_count, index1, index2):
    text_1 = list(text_pair_count[index1])
    text_pair_count[index1] = list(text_pair_count[index2])
    text_pair_count[index2] = list(text_1)
    for i in range(len(text_pair_count)):
        value_i1 = int(text_pair_count[i][index1])
        value_i2 = int(text_pair_count[i][index2])
        text_pair_count[i][index1] = value_i2
        text_pair_count[i][index2] = value_i1  