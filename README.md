![title.jpg](https://github.com/a-woodbury/RxID/blob/master/Images/RxID-1.jpg)

**Increasing Medication Safety with Deep Learning Image Recognition**

***This model is currently under development; please check back 05/29 for the latest metrics***

<pre>
Domain             : Computer Vision, Machine Learning
Sub-Domain         : Deep Learning, Image Recognition
Techniques         : Deep Convolutional Neural Network, ImageNet, Inception
Application        : Image Recognition, Image Classification, Medical Imaging
</pre>

<pre>
Contributors:
<a href=https://github.com/a-woodbury>Alphonso Woodbury</a>

</pre>

<pre>
Presentation: 
<a href=https://docs.google.com/presentation/d/e/2PACX-1vTJ6M3NdXdyokIdZT-mPS_Ke_d5NKyQmv7HWIxZ5hOrkwexsM331qzmdN7cBQ5PBvR20fsBACOMeMaM/pub?start=false&loop=false&delayms=3000>Google Slides</a>
</pre>

### Prompt:

ttps://www.hopkinsmedicine.org/news/media/releases/study_suggests_medical_errors_now_third_leading_cause_of_death_in_the_us


In contrast to the reduction in deaths per 100,000 from cancer and heart disease, deaths from medication continue to increase. These deaths are attributed to medication error, which presents in several ways. 

preventable medical errors are the third leading cause of death in the United States, and medication error is the primary contributor. About 2,460 people die each week due to a preventable medication error. 

- incorrect treatment plan
- incorrect dispensing
- incorrect administration, clinician
- incorrect administration, patient 
  - Omission
  - Self-medication
  - Incorrect dosage
  - Improper timing or sequence
  - Inaccurate knowledge

Where
prescriber
home
nursing home

choosing a medicine—irrational, inappropriate, and ineffective prescribing, underprescribing and overprescribing;
writing the prescription—prescription errors, including illegibility;
manufacturing the formulation to be used—wrong strength, contaminants or adulterants, wrong or misleading packaging;
dispensing the formulation—wrong drug, wrong formulation, wrong label;
administering or taking the drug—wrong dose, wrong route, wrong frequency, wrong duration;
monitoring therapy—failing to alter therapy when required, erroneous alteration.



https://ajph.aphapublications.org/doi/pdf/10.2105/AJPH.52.12.2018  

https://apps.who.int/iris/bitstream/handle/10665/252274/9789241511643-eng.pdf;jsessionid=BEA247A75DD2B5E427B5B75456948756?sequence=1  

https://www.cms.gov/Research-Statistics-Data-and-Systems/Statistics-Trends-and-Reports/NationalHealthExpendData/NHE-Fact-Sheet 

These are the types of errors related to pharmacy per the Institue of Medicine; “To increase patient safety, one of the major advances, in recent years, has been computerization. ”
https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2639900/ “Automation holds substantial promise, for improved safety, but error experts caution that all technology introduces the potential for new and different errors”

Adverse drug events cost the United States $37.6 billion each year, and approximately $17 billion of these costs are associated with preventable errors (Source: “To Error is Human: Building a Safer Health System,” a November 1999 report from the Institute of Medicine [IOM]).

Medication errors are the source of 5% of malpractice claims (Byron J. Bailey. “Medical Liability Issues: Where Malpractice Claims Come From and How to Survive Them,” July 2002). The average jury award for medication error cases is $636,844 (Jury Verdict Research, “2000 Current Award Trends in Personal Injury,” June 2001).






### Code
<pre>
GitHub Link      : <a href=Link>File Name </a>
</pre>

### Dataset
<pre>
Dataset Links    : <a href=https://www.nlm.nih.gov/databases/download/pill_image.html>NIH RxImage Portal</a>

This project uses 749 images, 15 classes from the NIH drug image dataset. The images are stored on an FTP server and can be queried and downloaded using the Data Collection notebook in this repository. 
</pre>

### Recommendations:

### Next Steps:
Firstly, we would like to implement transfer learning using VGG16 to execute feature extraction. This should provide a boost to our baseline performance. 


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
