#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 30 20:10:14 2021

@author: ali
"""

import sys
import random
    
    
def randomQueries(output_filename, query_size,dimension, numberRange):
    queries = []
    for i in range(query_size):
        left_point = ()
        right_point = ()
        for _ in range(dimension):
            random_left_data = random.randint(0, numberRange/2) # Let's have some space
            left_point += (random_left_data,)
            random_left_data = random.randint(random_left_data+1, numberRange)
            right_point += (random_left_data,)
            
        queries.append((left_point,right_point))
    
    f = open(output_filename,"w")
    f.write(str(len(queries)) + '\n')
    f.write(str(len(queries[0][0])) + '\n')
    
    for query in queries:
        f.write(' '.join([str(point_term) for point_term in query[0]]) + \
                ' ' + ' '.join([str(point_term) for point_term in query[1]]) + '\n')
        
    f.close()
    
    return queries
    
    

if __name__ == '__main__':
    if len(sys.argv) != 5:
        print("Usage: python randomQueries.py <OUTPUT_FILENAME.txt> <query_size> <dimension> <numberRange>")
    else:
        print("Input is " + sys.argv[1])
        randomQueries(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]))
