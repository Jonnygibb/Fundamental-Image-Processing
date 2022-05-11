# Fundamental-Image-Processing
Source code, solutions and report for WM391 Industrial Vision and Processing module. Various fundamental image processing functions are created in python for the spatial and frequency domain

In order to create a report without using word/libreoffice or other "What You See Is What You Get" (WYSIWYG) word processing suites, pandoc is used to generate a pdf report from a markdown file.

## Report Generation

Certain prerequisites are needed to generate a report in pandoc:
```console
sudo apt install pandoc

sudo apt install texlive-latex-recommended
```
Once these libraries are installed, it is possible to generate a pdf report as seen below:
```console
cd report/

pandoc report.md --bibliography references.bib --csl elsevier-harvard.csl --highlight-style mystyle.theme -o report.pdf
```