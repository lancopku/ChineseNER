import gzip

import urllib
import os
import random

from os.path import isfile

def load_data(filename):
    
    result = []
    with open(filename,'r') as f:
	    for line in f:
		    result.append(map(int, line.strip().split(' ')))
    return result
    
def load_emb(filename):
    
    result = []
    with open(filename,'r') as f:
	    for line in f:
		    result.append(map(float, line.split(' ')))
    return result

if __name__ == '__main__':
    
    result = load_data('example.txt')
    print (result)



	



