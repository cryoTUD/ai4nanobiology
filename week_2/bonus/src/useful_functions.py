
### copy of all usful functions from the exercise

import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from emmer.ndimage.filter.filter_utils import calculate_fourier_frequencies, tanh_filter

# useful functions
def rotate_image(image, angle):
    return ndimage.rotate(image, angle, reshape=False)

def SQD(image, image_stack):
    return ((image-image_stack)**2).sum((image_stack.ndim-2,image_stack.ndim-1))

def corr(image, image_stack):
    return (image*image_stack).mean((1,2))

# helpful plotting functions (you can also use this as a reference to make the plots yourself. See also https://matplotlib.org/stable/plot_types/index.html)
def plot_data(data, xlabel, ylabel, title):
    _, ax = plt.subplots()
    p = ax.plot(data[0], data[1])
    ax.set(xlabel, ylabel, title)
    ax.grid()
    return p

def plot_histogram(data, bins, xlabel, ylabel, title):
    _, ax = plt.subplots()
    hist = ax.hist(data, bins=bins)
    ax.set(xlabel, ylabel, title)
    ax.grid()
    return hist

def plot_scatter(data, xlabel, ylabel, title):
    _, ax = plt.subplots()
    scatter = ax.scatter(data[0], data[1])
    ax.set(xlabel, ylabel, title)
    ax.grid()
    return scatter

# useful functions
def low_pass_filter_image(im, cutoff=5, apix=1):
    """
    Returns a low-pass filter image from a tanh filter.
    """
    
    im_freq     = calculate_fourier_frequencies(im, apix=apix)
    im_filter   = tanh_filter(im_freq, cutoff);
    im_fft      = np.fft.rfftn(im)
    im_fft_filtered = im_fft * im_filter
    im_filtered = np.fft.irfftn(im_fft_filtered)
    return im_filtered

def make_circle(imsize, radius):
    im = np.zeros((imsize, imsize))
    for i in range(imsize):
        for j in range(imsize):
            if (i-imsize/2)**2 + (j-imsize/2)**2 < radius**2:
                im[i,j] = 1
    return im
