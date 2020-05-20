![title.jpg](https://github.com/a-woodbury/RxID/blob/master/Images/RxID-1.jpg)

**Increasing Medication Safety with Deep Learning Image Recognition**

### Problem

Despite increases in spending and tremendous advances in technology in the last several decades, medication errors and imjury/morbity from errors has contiued to increase. RxID attempts to use deep learning in an image recognition model to identify medications as to reduce the risk of medication error, and, ultimately, the human and financial cost of an Adverse Medical Event.

I accessed the NIH RxImage dataset to collect images for training a model. This dataset included a directory flat-file for analysis and referencing the images. The below notebooks provide guidance on accessing the directory and creating a model with the downloaded images.

If successful, this model has potential implementations discussed below in Recommendations. 

### Code
<pre>
GitHub Link      : <a href=https://github.com/a-woodbury/RxID/blob/master/RxID15_Data_Collection.ipynb>Data Collection Notebook </a>
GitHub Link      : <a href=https://github.com/a-woodbury/RxID/blob/master/RxID15_Modeling.ipynb>Modeling </a>
</pre>

### Dataset
<pre>
Dataset Links    : <a href=https://www.nlm.nih.gov/databases/download/pill_image.html>NIH RxImage Portal</a>

This project uses 749 images, 15 classes from the NIH drug image dataset. The images are stored on an FTP server and can be queried and downloaded using the Data Collection notebook in this repository. 
</pre>

### Results

The model achieved and Accuracy of 0.66 when classfying 15 drug classes over 100 epochs. 

![results.png](https://github.com/a-woodbury/RxID/blob/master/Images/CNN%204-Acc_Loss.png)


### Recommendations:

**Mail-order/Retail Pharmacy production**
- A supplemental check before releasing a prescription.

**Patient or Caregiver smartphone app**
- Visual impairment
- Drug change
- Recalls
- Mixed meds

**First Responder app** 

- Intervening in accidental or intentional poisoning

### Improving the current model

**Transfer Learning**
  - VGG16 candidate

**Additional augmentation; sciImage**
  - central_crop
  - block_out
  - contrast
 
### Next Steps:

Once the model accuracy is above ??%, I would like to use RxID15 to transfer learning to models for more medications

I have a vision of the model serving many diverse groups of patients, including those addicted to opioids. To help opiod users avoid ingesting contaminated medications, I would like to make a model for distinguishing authentic from counterfeit medications. 

A fascinating project in South Korea aimed to use image recognition on street recreational tablets with unique presses. Athough the constiution of such a drug could not be validated via image recognition, it could assist recreation drug users in avoid detrimental effects of potentialy tainted products. 

### Presentation

<pre>
<a href=https://github.com/a-woodbury/RxID/blob/master/Presentation/RxID.pdf>Slide Deck</a>
</pre>

### Tools / Libraries
<pre>
Languages               : Python
Tools/IDE               : Anaconda, Colab
Libraries               : Tensorflow, Keras, imagio, PIL, ftplib
</pre>

### Dates
<pre>
Duration                : May 2020
Last Update             : 05.20.2020
</pre>

<pre>
Domain             : Computer Vision, Machine Learning
Sub-Domain         : Deep Learning, Image Recognition
Techniques         : Deep Convolutional Neural Network, 
Application        : Image Recognition, Image Classification
</pre>
