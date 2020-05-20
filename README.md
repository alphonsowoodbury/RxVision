![title.jpg](https://github.com/a-woodbury/RxID/blob/master/Images/RxID-1.jpg)

**Increasing Medication Safety with Deep Learning Image Recognition**

<pre>
Domain             : Computer Vision, Machine Learning
Sub-Domain         : Deep Learning, Image Recognition
Techniques         : Deep Convolutional Neural Network, 
Application        : Image Recognition, Image Classification
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

### Code
<pre>
GitHub Link      : <a href=Link>Data Collection Notebook </a>
GitHub Link      : <a href=Link>Modeling </a>
</pre>

### Dataset
<pre>
Dataset Links    : <a href=https://www.nlm.nih.gov/databases/download/pill_image.html>NIH RxImage Portal</a>

This project uses 749 images, 15 classes from the NIH drug image dataset. The images are stored on an FTP server and can be queried and downloaded using the Data Collection notebook in this repository. 
</pre>

<pre>
Presentation: 
<a href=https://docs.google.com/presentation/d/e/2PACX-1vTJ6M3NdXdyokIdZT-mPS_Ke_d5NKyQmv7HWIxZ5hOrkwexsM331qzmdN7cBQ5PBvR20fsBACOMeMaM/pub?start=false&loop=false&delayms=3000>Google Slides</a>
</pre>

### Results

THe model achieved and Accuracy of 0.66 when classfying 15 drug classes over 100 epochs. 

![results.png](https://github.com/a-woodbury/RxID/blob/master/Images/CNN%204-Acc_Loss.png)


### Recommendations:

**Mail-order/Retail Pharmacy production**
- A supplemental check before releasing a prescription.

**Patient or Caregiver smartphone app**
  - Visual impairment
  - Drug change
  - Recalls
  - Mixed meds

**First Responder app** for intervening in accidental or intentional poisoning

### Improvement

 - Transfer Learning
 - More classes
 
### Next Steps:
To improve model performance and utility, we aim to increase accuracy and the number of classes (medications). 

**Performance**
- Transfer Learning

**Utility**
We have identified the most common medications dispensed in the us by ingredient, but will need to train our model on as many of the manufacturer's tablets/capsules as possible to increase the usefulness of the model.





