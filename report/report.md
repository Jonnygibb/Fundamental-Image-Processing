---
title: Investigation and Analysis of Fundamental Image Processing Concepts
author: Jonathan Gibbons
date: 10th May 2022
toc: true
numbersections: true
geometry: margin=2.5cm
urlcolor: blue
bibliography: references.bib
csl: elsevier-harvard.csl
header-includes: |
    \usepackage{fancyhdr}
    \pagestyle{fancy}
    \lfoot{Draft Prepared: 3rd May 2022}
    \rfoot{Page \thepage}
---

\newpage{}

# Abstract

This technical report seeks to discuss whether, using only video cameras, autonomous driving can be achieved. The implementation of fundamental image processing techniques are later analysed with key findings summarised.

# Introduction

# Literature Review

# Image Processing Implementation
The fundamental functions to perform image processing in the spatial and frequency domain were implemented alongside this report. A Gaussian blur was used as the example kernel.

## Summary and Analysis
Noteable discoveries in computational time, padding types and complexity are summarised below.

### Increased Efficiency for kernel generation
During creation of a spatial domain gaussian kernel of size 6(sigma) + 1, the kernel is square meaning it's possible to create the kernel without a distance matrix as shown in the code snippet below.

```python
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
```

Generating the kernel using this method lowers computation time because the gaussian formula, for a kernel of side lengths M, is applied to only M values rather than M^2^ values. The overall complexity is reduced to O(M + M^2^) rather than the distance matrix solution that has complexity O(2M^2^).

### Use of Padding in the Frequency Domain
In this implementation of a gaussian blur in the frequency domain, the image and gaussian kernal are made to be the same size. However for a kernel of set size, the image and/or kernel might have to be zero padded. Padding increases the total time that signals are recorded and therefore collects more of the frequencies in the center [@7676243]. Whilst the accuracy of the forier transform is not necessarily increased, it should be noted that the added padding will increase computation time.

### Computational Time of Spatial vs Frequency Domain
During analysis, the frequency domain was found to be significantly faster in computation time. For the same colour image over a range of cutoff frequencies, the frequency domain proved consistently to be 5 times faster on average. This increased efficiency can be seen in greater detail when using images of larger resolutions [@Gonzalez2020-zx]. The car image in the test notebook is up to 7 times faster in the frequency domain than the spatial domain. The relationship can be explained by the complexity of spacial filtering vs frequency filtering. In a simplified implementation, with image size M by N and kernel size K by L, the spatial domain has complexity O(mnkl) whereas the frequency domain has smaller computational complexity O(mn log(mn) + mn) [@Shreekanth2017].

### Change in spatial kernel size vs computation time

# Conclusion

# References