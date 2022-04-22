#!/usr/bin/env python
# coding: utf-8


import numpy as np
from numpy import pi, exp
from pandas import cut
from skimage import io, img_as_float32
from skimage.transform import rescale
import matplotlib.pyplot as plt



def ideal_LP(width,k_size):
    
    
    """
	The function will take two arguments (width, k_size), the width of the pass band and the size of the kernel, and return the ideal_LP_kernel array matching the input k_size
	
	"""
	#########Your code here#############
    raise NotImplementedError('Please implement `ideal_LP` function in `A1_template.py`' )

    
	
    return ideal_LP_kernel

#%%


def gaussLP_2D_space(cutoff_sigma, scale=1):
    """
    Implement a gaussian low pass filter function which outputs a spatial gaussian kernel
    of size k x k, where k = (6*cutoff_sigma)+1    

    Parameters
    ----------
    cutoff_sigma : int 
        The standard deviation of the Gaussian distribution.
    scale : int
        scale parameter of the Gaussian distribution.

    Returns
    -------
    gauss_spatial : array of size k x k
        Kernel to convolve the image with.

    """
	#########Your code here#############
    k_mid_point = 3*np.ceil(cutoff_sigma)        # set to 3 for computational ease, approx 95% energy
                                                 # three times would be better
    k = (6*cutoff_sigma) + 1

    ax = np.linspace(-(k - 1) / 2., (k - 1) / 2., k)
    gauss = np.exp(-0.5 * np.square(ax) / np.square(cutoff_sigma))
    kernel = np.outer(gauss, gauss)
    gauss_spatial = kernel / np.sum(kernel)

    #raise NotImplementedError('Please implement `gaussLP_2D_space` function in `A1_template.py`' )
    
    return gauss_spatial


#%%



def spatial_dom_filter(image, filter):
    """
    The function will take two arguments (image, kernel), numpy arrays containing the image and the kernel respectively, and return the filtered_image array that matches the size of the input image. The filtered_image is a result of the convolution of the input image and the filter. Also, the function only needs to deal with square kernels of odd integer size.  
	
    """
    assert filter.shape[0] % 2 == 1
    assert filter.shape[1] % 2 == 1

    ###YOUR CODE HERE ###
    # Save the width of the filter kernel
    kernel_width = filter.shape[0]

    # Create array of zeroes to store filtered values
    filtered_image = np.zeros(shape=(image.shape[0], image.shape[1], image.shape[2]))
    # Added padding to the image to account for kernel overlap
    padding = int((kernel_width - 1) / 2)
    padded_image = np.pad(image,
                   ((padding, padding), (padding, padding), (0,0)))

    # Iterate through the colour channels of the image
    for z in range(image.shape[-1]):
        channel = padded_image[:,:,z]
        for x in range(image.shape[0]):
            for y in range(image.shape[1]):
                sample = channel[x:x+padding, y:y+padding]
                filtered_image[x, y, z] = np.sum(np.multiply(sample, filter))
    
    return filtered_image



#%%
def gaussLP_2D_freq(cutoff_freq, k_size, scale_parmeter = 1):
    
    """
	The function will take three arguments cutoff_freq, k_size, scale_parmeter, and return the gauss_2d array of size k_size, representing the Gaussian filter in the frequency domain.
	"""
	
	########Your code here#############
    raise NotImplementedError('Please implement `gaussLP_2D_freq` function in `A1_template.py`' )
    return gauss_2d


#%%

def freqency_dom_filter(image,filter):
    """
	The function will take  arguments (image, filter), and return the filtered_image array"""	
	########Your code here#############
    #    
    raise NotImplementedError('Please implement `freqency_dom_filter` function in `A1_template.py`' )
    return filtered_image