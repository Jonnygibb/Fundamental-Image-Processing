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

# Literature Review

# Image Processing Implementation
The fundamental functions to perform image processing in the spatial and frequency domain were implemented alongside this report. A Gaussian blur was used as the example kernel.

## Summary and Analysis
Noteable discoveries in computational time, padding types and complexity are summarised below.

### Computational Time of Spatial vs Frequency Domain
During analysis, the spatial domain was found to be significantly slower in computation time. For the same colour image over a range of cutoff frequencies, the frequency domain proved consistently to be 5 times faster on average. This increased efficiency can be seen in greater detail when using images of larger resolutions, as seen in the test notebook and in the below figure. 

# References