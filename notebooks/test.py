from math import log
from math import isclose
import numpy
import random
import re
from utils import *

alpha_list = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z',' ']

def test_num_to_text_and_back(char_text):
    num_text = char_to_num(char_text)
    new_char_text = num_to_char(num_text)
    assert (char_text == new_char_text), 'should be the same'
    print('test passed: num_to_text and text_to_num')
    
def encode_decode_caesar(plaintext, key):
    ciphertext = encode_caesar_cipher(plaintext, key)
    new_plaintext = decode_caesar_cipher(ciphertext, key)
    assert (plaintext == new_plaintext), 'should be the same'
    print('test passed: encode_caesar and decode_caesar')
    
def encode_decode_aris(plaintext, num_key):
    ciphertext = encode_aris(plaintext, num_key)
    new_plaintext = decode_aris(ciphertext, num_key)
    assert (plaintext == new_plaintext), 'should be the same'
    print('test passed: encode_aris and decode_aris')


def test_calc_change(test_text, changed_text, l1, l2):
    text_letter_count = count_letters(test_text)
    text_pair_count = count_letter_pairs(test_text)
    one = find_pair_log_likelihood(test_text, bible_letter_percent, bible_matrix)
    two = find_pair_log_likelihood(changed_text, bible_letter_percent, bible_matrix)
    diff = (two - one)
    start_key = list(alpha_list)
    num_key = char_to_num(start_key)
    index1 = alpha_list.index("h")
    index2 = alpha_list.index("l")
    faster_calc = (calculate_log_likelihood_change(test_text, text_pair_count, bible_letter_percent, bible_matrix, index1, index2, num_key))
    assert isclose(diff, faster_calc), 'should be equal'
    print("test passed: calc_change")

def test_switch_letters():
    test_text = 'hallo how era you'
    changed_text = switch_letters(test_text, 'a', 'e')
    assert (''.join(changed_text) == 'hello how are you'), 'should be the same'
    print('test passed: switch_letters')
    
def test_switch_row_and_columns():
    test_matrix = [[1,2,3],[4,5,6],[7,8,9]]
    switch_row_and_columns(test_matrix, 1, 2)
    assert (test_matrix == [[1,3,2],[7,9,8],[4,6,5]])
    print('test passed: switch_row_and_columns')


bible_letters = make_letters('../data/bible.txt')
bible_letter_count = count_letters(bible_letters)
bible_letter_percent = normalize_counts_no_spaces(bible_letter_count)
bible_pair_count = count_letter_pairs(bible_letters)
bible_matrix = compute_transition_matrix(bible_pair_count, 0.5)

tt = 'hello'
ct = 'lehho'
l1 = 'h'
l2 = 'l'
                   
test_calc_change(tt, ct, l1, l2)
test_switch_letters()
test_switch_row_and_columns()
