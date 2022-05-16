---
title: Investigation and Analysis of Fundamental Image Processing Concepts
author: u1921415
date: 15th May 2022
toc: true
numbersections: true
geometry: margin=2.5cm
urlcolor: blue
bibliography: references.bib
csl: elsevier-harvard.csl
header-includes: |
    \usepackage{fancyhdr}
    \pagestyle{fancy}
    \lfoot{15th May 2022}
    \rfoot{Page \thepage}
---

\newpage{}

# Abstract
This technical report seeks to discuss whether, using only video cameras, autonomous driving can be achieved. The implementation of fundamental image processing techniques are later analysed with key findings summarised.

# Introduction
Digital image processing is the analysis of images using algorithms. Any field that requires understanding of the physical world would be benefit from some form of image processing. The medical, automotive and aerospace are just a handful of domains that utilise image processing to gain deeper understanding from the images they produce. 

# Literature Review

## Can autonomous driving be achieved with only video cameras in the visible and infrared spectrum?
The choice of sensors when it comes to autonomous driving systems comes down to visible light cameras, infrared cameras, RADAR and LIDAR. Visible light cameras excel at tasks such as lane departure warning systems. Image processing can be applied to an actual view of the road lines either side of the car [@Cario2017-kl]. Despite this, visible light cameras struggle in foggy, dark and wet weather conditions [@6338748] making them inappropriate for tasks such as automated cruise control, where the car maintains a safe distance to the one ahead, at night as its poor visibility could lead to a collision.

Infrared cameras are more than capable at working in poor visibility and at long distances and high speeds [@8806152]. They detect heat energy radiated from objects and use that distinction to identify objects. This allows them to work in the conditions where visible light cameras fail making systems like pedestrian detection a reality. With that said, many pedestrian detection and automatic breaking systems also rely on LiDAR to provide more accurate distance detection [@Deshpande2017-ir]. Despite the additional cost of a LiDAR sensor, the safety of automated driving systems cannot be entirely guaranteed with only thermal imaging.

An established sensor used in autonomous driving systems is that of RADAR. Being cheap, long range and reliable in many conditions, RADAR has already seen use in ADAS systems. Using a similar Time of flight system to LiDAR, where the time taken for a signal to return from an object calculates distance, RADAR offers a versatile method of understanding a physical space. However the downside of low resolution makes RADAR unable to be a one-size-fits-all solution to autonomous driving [@petit_2022].

While sensors of visible and infrared light, when paired together, manage to handle many aspects of autonomous driving, both sensors have limitations that leave them vulnerable in certain situations. To reach truly autonomous level 6 vehicles, a combination of sensors across the electromagnetic spectrum including LiDAR and RADAR must be involved to make sure that the safety of those inside and outside the vehicle is increased. LiDAR sensors are still rather expensive so the automotive industry may have to wait before autonomous vehicles with a full roster of sensors can be made commercially.

# Image Processing Implementation
The fundamental functions to perform image processing in the spatial and frequency domain were implemented alongside this report. A Gaussian blur was used as the example kernel.

## Summary and Analysis
Notable discoveries in computational time, padding types and complexity are summarised below.

### Effect of changing cutoff frequency and scale factor in frequency domain
Increasing the standard deviation of the Gaussian creates a more drastically blurred image. This is because the formula $f0 = 1/2*pi*sigma$ is used, so as standard deviation increases, the cutoff frequency decreases meaning more of the high frequency components are filtered out of the image. The effect of increasing the scale factor upwards from 1 results in a brightened image. By multiplying the values of the image array by a scale factor greater than one makes each pixel value greater resulting in the image trending towards an all white image.

### Increased Efficiency for kernel generation
When investigating the creation of a spatial domain Gaussian kernel, two methods were found. The kernel is square meaning it's possible to create the kernel without a distance matrix as shown in the code snippet below.

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

Generating the kernel using this method has the lowest computation time because the Gaussian formula, for a kernel of side lengths M, is applied to only M values rather than M^2^ values. The overall complexity is reduced to O(1). The distance matrix solution can be seen below and demonstrates the larger complexity of O(M^2^).

```python
k = (6*cutoff_sigma) + 1
distance = np.zeros((k, k))
# Iterate over each cell of the zero matrix
for row in range(k):
    for column in range(k):
        # Set each cell equal to the pythagorian distance from the center of the matrix
        distance[row, column] = np.sqrt((row - rows / 2)**2 + (column - columns / 2)**2)
        # Apply the gaussian distribution formula to the distance matrix
        distance[row, column] = np.exp(-((distance[row,column]**2)/(2*cutoff_freq**2)))
gauss_spatial = distance
return gauss_spatial
```

### Use of Padding in the Frequency Domain
Padding then Forier transforming an image increases the total time that signals are recorded [@7676243]. This added time improves the frequency resolution and interpolation of the image in the Forier domain. The observed result in this implementation is that the produced blurred image appears the same with or without padding applied however the intensity of the resultant image is greater without padding than with. This effect can be seen in the test notebook.

### Computational Time of Spatial vs Frequency Domain
During analysis, the frequency domain was found to be significantly faster in computation time. For the same colour image over a range of cutoff frequencies, the frequency domain proved consistently to be 5 times faster on average. This increased efficiency can be seen in greater detail when using images of larger resolutions [@Gonzalez2020-zx]. The car image in the test notebook is up to 7 times faster in the frequency domain than the spatial domain. The relationship can be explained by the complexity of spacial filtering vs frequency filtering. In a simplified implementation, with image size M by N and kernel size K by L, the spatial domain has complexity O(mnkl) whereas the frequency domain has smaller computational complexity O(mn log(mn) + mn) [@Shreekanth2017].

# Conclusion
Having researched the commercial uses of image processing and its ability to change the way we travel as well as fundamentals of image processing algorithms, its clear that the field will continue to progress as sensors and computational power increases. 

Word Count: 1214 words

# References