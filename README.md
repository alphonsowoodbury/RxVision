![title.jpg](https://github.com/a-woodbury/RxID/blob/master/Images/RxID-1.jpg)

**Increasing Medication Safety with Deep Learning Image Recognition**

### Overview

Despite increases in healthcare spending and tremendous advances in technology in the last several decades, medication errors and injury/morbity from errors have continued to increase. The abuse of opiods have certainly contributed to mortality from medication overdoses, but the vast majority of Adverse Drug Events occur from patients taking medications as prescribed. The RxID project attempts to use deep learning in an image recognition model to identify medications and propose deployment solutions to reduce the risk and occurence of medication error, and, ultimately, the human and financial cost of an Adverse Medical Reaction.

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

Americans are taking more medications than ever. In 2019, more than 4 billion prescriptions were dispensed in retail and mail-order  pharmacies at a cost of $551 billion[kff]. This is in sharp contrast to the 3 billion prescriptions and $117 billion spent in 2000. Across all demographics, 89% of Americans take at least one daily medication, while 29% are taking 5 or more. Per capita, we've gone from 10.8 presctions in 2000 to in 11.9 in 2019. 

These data are skewed when chronic, costly conditions are considered. For example, patients treated for diabetes take 4 times as many medications as those not recieing diabetes treatment[gw]. Another major driver of prescription use increase is the aging population; about 52 million Americans are of age 65 or older and taking the lions share of medications.

When considering this growth, it becomes more clear why medical/medication error would be identified as the third leading cause of death in the US in 2016. As patients take more medications, the risk of interactions, side effects, and errors increase as well. The FDA provides guidance for defining, monitoring, and enforcing practices to reduce medical and medication errors. 

> **Medication Error:** Any preventable event that may cause or lead to inappropriate medication use or patient harm while the medication is in the control of the healthcare professional, patient, or consumer.
> 
> **Adverse Medical Event (ADE):** Any abnormal sign, symptom or laboratory test, or any syndromic combination of such abnormalities, any untoward or unplanned occurrence (e.g. an accident or unplanned pregnancy), or any unexpected deterioration in a concurrent illness
> 
> **Adverse Medical Reaction (ADR):** an appreciably harmful or unpleasant reaction, resulting from an intervention related to the use of a medicinal product. 

<p>
  <img src=https://github.com/a-woodbury/RxID/blob/master/Images/ADE.png title="ADE venn" class="center"/>
  <figcaption>Fig.1 - Medication Errors, Events, and Reactions.</figcaption>
</p>

<p align="center">
 <figure>
   <img src=https://github.com/a-woodbury/RxID/blob/master/Images/ADE.png title="ADE venn"/>
 </p>
   <figcaption>Fig.1 - Medication Errors, Events, and Reactions.</figcaption>
 </figure>


When an ADE or ADR is identified, the course of action is clinical intervention with the patient and, if occurring in a clinical setting, reporting and correction action with the offending agency. However, as seen in the above graphic, not all ADEs and ADRs are caused by medication error, and not all medication errors lead to an ADE or ADR. We want to focus on reducting medication errors at the pharmacy and patient level as much as possible. 

#### Progress

What has been done to reduce medication errors so far

#### Pill Identification

what projects for pill ID; why no implementation?



## Data

This project uses 749 images, 15 classes from the NIH drug image dataset. The images are stored on an FTP server and can be queried and downloaded using the Data Collection notebook in this repository.

**Reproduction:** The code needed to request and download images from the NIH server are in the [Data Collection]() notebook. Additional information on this process can be found in the [Recreating the Model Guide]().

***Note:*** In the near future, the NIH dataset will be available from Google Cloud Platform. Videos and corrupt files will be excluded, and all files will be in a smaller format JPG file for faster download and processing. The notebooks will be updated to reflect this change in data source. 

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
