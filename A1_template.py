#!/usr/bin/env python
# coding: utf-8


import numpy as np
from numpy import pi
import matplotlib.pyplot as plt


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
    # Calcualte k from input sigma value  
    k = (6*cutoff_sigma) + 1

    # Evenly distribute a 1D vector from negative half (k-1)/2
    # to positive half (k-1)/2.
    # E.g when k = 5, array = [-2, -1, 0, 1, 2].
    # This array represents the distances from the center of the kernel
    # to the xth or yth element of the kernel.
    dxy = np.linspace(-(k - 1) / 2., (k - 1) / 2., k)
    # Apply the gaussian distribution formula to the 1D vector.
    # Square the distances from the center of the kernel and multiply
    # by -0.5 then divide by the square of cutoff sigma as per the formula.
    # Create a vector that is e to the power of the previous steps.
    gauss = np.exp(-0.5 * np.square(dxy) / np.square(cutoff_sigma))
    # Use the outer product of vectors in order to multiply the distributed
    # vector by itself to create a 2D matrix that maintains the gaussian
    # distribution.
    kernel = np.outer(gauss, gauss)
    # Apply the scale factor to the kernel
    scaled_kernel = np.dot(scale, kernel)
    # Divide the scaled kernel by it's sum to normalise the values to sum
    # to 1.
    gauss_spatial = scaled_kernel / np.sum(scaled_kernel)
    return gauss_spatial


#%%



def spatial_dom_filter(image, filter):
    """
    The function will take two arguments (image, kernel), numpy arrays containing the image and the kernel respectively,
    and return the filtered_image array that matches the size of the input image. The filtered_image is a result of the
    convolution of the input image and the filter. Also, the function only needs to deal with square kernels of odd
    integer size.  
    """
    assert filter.shape[0] % 2 == 1
    assert filter.shape[1] % 2 == 1

    # Save the width of the filter kernel
    kernel_width = filter.shape[0]

    # Check whether the image is grayscale
    if(len(image.shape) < 3):
        # Create array of zeroes to store filtered values
        filtered_image = np.zeros(shape=(image.shape[0], image.shape[1]))
        # Added padding to the image to account for kernel overlap
        padding = int((kernel_width - 1) / 2)
        padded_image = np.pad(image,
                        ((padding, padding), (padding, padding)))
        # Iterate over x&y coordinates of the image array
        for x in range(image.shape[0]):
            for y in range(image.shape[1]):
                # Take a sample of the image the same size as the filter
                sample = padded_image[x:x+kernel_width, y:y+kernel_width]
                # Convolve the image and filter together to apply the blur
                filtered_image[x, y] = np.sum(np.multiply(sample, filter))
    # Check if the image is colour
    elif(len(image.shape) == 3):
        # Create array of zeroes to store filtered values
        filtered_image = np.zeros(shape=(image.shape[0], image.shape[1], image.shape[2]))
        # Added padding to the image to account for kernel overlap
        padding = int((kernel_width - 1) / 2)
        padded_image = np.pad(image,
                    ((padding, padding), (padding, padding), (0,0)))

        # Iterate through the colour channels of the image
        for z in range(image.shape[-1]):
            # Seperate the image into its RGB colour channels
            channel = padded_image[:,:,z]
            # Iterate over x&y coordinates of the image colour channel
            for x in range(image.shape[0]):
                for y in range(image.shape[1]):
                    # Take a sample of the image the same size as the filter
                    sample = channel[x:x+kernel_width, y:y+kernel_width]
                    # Convolve the image and filter together
                    filtered_image[x, y, z] = np.sum(np.multiply(sample, filter))
    else:
        raise Exception("Image type not supported")
    
    return filtered_image



#%%
def gaussLP_2D_freq(cutoff_freq, k_size, scale_parameter = 1):
    """
	The function will take three arguments cutoff_freq, k_size, scale_parmeter,
    and return the gauss_2d array of size k_size, representing the Gaussian filter
    in the frequency domain.
	"""
    # Change the cutoff frequency using the 1/2*pi*sigma relationship between
    # spatial and frequency domain
    cutoff_freq = ((k_size[0] + k_size[1]) / 2 ) / (2 * pi * cutoff_freq)

    # Simplify the k_size into row number and column number
    rows, columns  = k_size[0], k_size[1]
    # Create a matrix of zeros of equal size to the image/filter
    distance_matrix = np.zeros((rows, columns))
    # Iterate over each cell of the zero matrix
    for row in range(rows):
        for column in range(columns):
            # Set each cell equal to the pythagorian distance from the center of the matrix
            distance_matrix[row, column] = np.sqrt((row - rows / 2)**2 + (column - columns / 2)**2)
            distance_matrix[row,column] = scale_parameter * np.exp(-((distance_matrix[row,column]**2)/(2*cutoff_freq**2)))
    gauss_2d = distance_matrix
    plt.subplot(121),plt.imshow(gauss_2d)
    
    return gauss_2d


#%%

def freqency_dom_filter(image,filter):
    """
	The function will take  arguments (image, filter), and return the filtered_image array
    """	
    # Shift the zero frequency components to the center of the image
    filter = np.fft.ifftshift(filter)
    # Check whether image is grascale
    if(len(image.shape) < 3):
        # Make a zero array the same size as the image
        filtered_image = np.zeros(shape=(image.shape[0], image.shape[1]))
        # Apply the forier transform to the image to convert it into the frequency domain
        freq_image = np.fft.fft2(image)
        # Convolve the image and the filter together then inverse the forier tansform
        # to convert back to the spacial domain
        filtered_image = np.fft.ifft2(np.multiply(freq_image, filter))
        # Consider only the real components of the image
        filtered_image = np.real(filtered_image)
    # Check whether image is colour
    elif(len(image.shape) == 3):
        # Make a zero array the same size as the image
        filtered_image = np.zeros(shape=(image.shape[0], image.shape[1], image.shape[2]))
        # Iterate through the colour channels of the image
        for z in range(image.shape[-1]):
            # Work on a colour channel by colour channel basis
            channel = image[:,:,z]
            # Convert specific channel to the frequency domain
            freq_channel = np.fft.fft2(channel)
            # Convolve the channel and the filter together in the frequency domain
            # then convert back to the spatial domain. Save the channel to the 
            # appropriate filtered_image channel
            filtered_image[:,:,z] = np.fft.ifft2(np.multiply(freq_channel, filter))
    else:
        raise Exception("Image type not supported")
    return filtered_image