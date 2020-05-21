![title.jpg](https://github.com/a-woodbury/RxID/blob/master/Images/RxID-1.jpg)

**Increasing Medication Safety with Deep Learning Image Recognition**

### Overview

Despite increases in spending and tremendous advances in technology in the last several decades, medication errors and imjury/morbity from errors has contiued to increase. RxID attempts to use deep learning in an image recognition model to identify medications as to reduce the risk of medication error, and, ultimately, the human and financial cost of an Adverse Medical Event.

I accessed the NIH RxImage dataset to collect images for training a model. This dataset included a directory flat-file for analysis and referencing the images. The below notebooks provide guidance on accessing the directory and creating a model with the downloaded images.

If successful, this model has potential implementations discussed below in Recommendations. 

### Repository Navigation
<pre>
Technical Notebook : <a href=Link>Technical Notebook </a>
Other Notebooks    : <a href=https://github.com/a-woodbury/RxID/blob/master/RxID15_Modeling.ipynb>Modeling</a>, <a href=https://github.com/a-woodbury/RxID/blob/master/RxID15_Data_Collection.ipynb>Data Collection Notebook </a>
Dataset Links      : <a href=https://www.nlm.nih.gov/databases/download/pill_image.html>NIH RxImage Portal</a>, <a href=Link>GCP Bucket</a>
Presentation       : <a href=https://github.com/a-woodbury/RxID/blob/master/Presentation/RxID.pdf>Slide Deck</a>, <a href=https://docs.google.com/presentation/d/1f2bLza9GFhIXUAMudNsb00RTpHAwg5JegGIw2i2Jg8A/edit?usp=sharing>Google Slides</a>
Other              : <a href=Link>Recreating the Model Guide</a>, <a href=Link>Drug Classes</a>
</pre>

### ReadME Navigation

[Problem](https://github.com/a-woodbury/RxID/blob/master/README.md#problem) - 
[Data](https://github.com/a-woodbury/RxID#data) -
[Model](https://github.com/a-woodbury/RxID#model) -
[Results](https://github.com/a-woodbury/RxID#results) - 
[Recommendations](https://github.com/a-woodbury/RxID#recommendations) - 
[Future](https://github.com/a-woodbury/RxID#future) - 
[Project Info](https://github.com/a-woodbury/RxID#project-info) -
[Works Cited](https://github.com/a-woodbury/RxID#works-cited)

## Problem

<p align="center">
  <img src=https://github.com/a-woodbury/RxID/blob/master/Images/ADE.png title="ADE venn"/>
</p>

## Data

This project uses 749 images, 15 classes from the NIH drug image dataset. The images are stored on an FTP server and can be queried and downloaded using the Data Collection notebook in this repository.

**Reproduction:** The code needed to request and download images from the NIH server are in the [Data Collection]() notebook. Additional information on this process can be found in the [Recreating the Model Guide]().

In the near future, the NIH dataset will be available from Google Cloud Platform. Videos and corrupt files will be excluded, and all files will be in a smaller format JPG file for faster download and processing. 

## Model

## Results

### Improving the current model

## Recommendations:

## Future

![RxID%Future.png](https://github.com/a-woodbury/RxID/blob/master/Images/RxID%20Future.png)
Once the model accuracy is above ??%, I would like to use RxID15 to transfer learning to models for more medications

I have a vision of the model serving many diverse groups of patients, including those addicted to opioids. To help opiod users avoid ingesting contaminated medications, I would like to make a model for distinguishing authentic from counterfeit medications. 

A fascinating project in South Korea aimed to use image recognition on street recreational tablets with unique presses. Athough the constiution of such a drug could not be validated via image recognition, it could assist recreation drug users in avoid detrimental effects of potentialy tainted products. 

## Project Info

<pre>
Contributors : <a href=https://github.com/a-woodbury>Alphonso Woodbury</a>
</pre>

<pre>
Languages    : Python
Tools/IDE    : Anaconda, Colab
Libraries    : Tensorflow, Keras, imagio, PIL, ftplib
</pre>

<pre>
Duration     : May 2020
Last Update  : 05.20.2020
</pre>

<pre>
Domain       : Computer Vision, Machine Learning
Sub-Domain   : Deep Learning, Image Recognition
Techniques   : Deep Convolutional Neural Network, 
Application  : Image Recognition, Image Classification
</pre>

## Works Cited
