#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 30 20:10:14 2021

@author: ali
"""

import numpy as np
import sys
    
def randomPoints(output_filename, numPoints, dimension, numberRange):
    points = np.random.randint(numberRange, size=(numPoints, dimension))
    np.savetxt(output_filename,points, fmt='%d', header=str(points.shape[0]) \
               +"\n" + str(points.shape[1]), comments='')
    

if __name__ == '__main__':
    if len(sys.argv) != 5:
        print("Usage: python randomPoints.py <OUTPUT_FILENAME.txt> <numPoints> <dimension> <numberRange>")
    else:
        print("Input is " + sys.argv[1])
        randomPoints(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]))
