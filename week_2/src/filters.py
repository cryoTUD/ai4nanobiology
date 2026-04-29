# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 13:39:21 2020

@author: abharadwaj1
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as image
from scipy.signal import convolve2d

def get_sobel_x(size):
    if size == 3:
        return np.matrix([-1,0,1]).transpose() * np.matrix([1,2,1])
    else:
        transform_matrix = np.matrix([1,2,1]).transpose() * np.matrix([1,2,1])
        return convolve2d(transform_matrix,get_sobel_x(size-2))

def get_sobel_y(size):
    if size == 3:
        return np.matrix([1,2,1]).transpose() * np.matrix([-1,0,1])
    else:
        transform_matrix = np.matrix([1,2,1]).transpose() * np.matrix([1,2,1])
        return convolve2d(transform_matrix,get_sobel_y(size-2))
    
def sobel(size=3,ax='xy'):
    if ax=='x':
        return get_sobel_x(size)
    elif ax=='y':
        return get_sobel_y(size)
    else:
        return get_sobel_x(size),get_sobel_y(size)
    
    
def identity(size=3):
    output = np.zeros((size,size))
    output[size//2,size//2] = 1
    return output

def edge_detection(size=3):
    output = np.zeros((size,size))
    output[:,size//2] = 1
    output[size//2,:] = 1
    output[size//2,size//2] = -(2*size-2)
    return output

def sharpen(size=3):
    '''
    output = np.ones((size,size)) * -1
    output[size//2,size//2] = size**2
    '''
    output = identity(size) - edge_detection(size)
    #identity(size) - sobe    
    return output

def box_blur(size=3):
    output = np.ones((size,size)) * 1/size**2
    return output

def gaussian_blur(size=3):
    std = size/4
    xo,yo = size//2,size//2
    y,x = np.ogrid[0:size,0:size]
    xv,yv = np.meshgrid(x,y)
    gauss_output = (np.exp(-1*( ((xv-xo)**2 / (2*std**2)) + ( (yv-yo)**2 / (2*std**2) ))))*1/(2*np.pi*std**2)
    return gauss_output


    
    
    
