---
title: Investigation into the use of Convolutional Neural Networks for the purpose of image exposure correction
author: u1921415
date: 12th June 2022
toc: true
numbersections: true
geometry: margin=2.5cm
urlcolor: blue
bibliography: references.bib
csl: elsevier-harvard.csl
header-includes: |
    \usepackage{fancyhdr}
    \pagestyle{fancy}
    \lfoot{12th June 2022}
    \rfoot{Page \thepage}
---

\newpage{}

# Abstract
This report examines the implementation of a Convolutional Neural Network for image exposure correction. A Generative Adversarial Network (GAN) style network architecture  is used and it's loss functions analysed.

# Introduction
Image exposure correction is a typical activity for photographers to adjust the light levels present in their photos. The exposure is set by the quantity of light let in during image capture. Adjustments are often made using image processing software such as adobe lightroom after the images are taken by the photographer. The image processing is a manual process that the photographer must perform for every image that they capture. This report explores whether a Generative Adversarial Network (GAN) could be used in order to generate well exposed images from a given under or over exposed image.

# Network architecture implemented
To achieve image translation from an image of poor exposure to one of more professionally exposed quality, a Generative Adversarial Network archtitecture was used. A GAN is a type of Convolutional Neural Network architecture with a unique construction. The generative part of the GAN structure is a generator model, whose role is to produce an output image for a given inputted image. On the other side of the GAN structure is the discriminator model which is trained to determine whether an image is real as in straight from the dataset of ground truth images, or fake and has been created by the generator. This architecture allows the discriminator to critique the generated images. The criticism is feedback to the generator model which reacts by updating it's weights to attempt to create more lifelike images after every epoch.





# Input output manipulations

# Loss function/s used

# Training and testing process followed

# Model evaluation metric/s used

# Analysis and conclusions

# Description of possible alternative approaches

# References