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

These data are skewed when chronic, costly conditions are considered. Having multiple chronic conditions leads to not only more medications but more contraindicated medicines and treatments for side-effects. For example, patients treated for diabetes take 4 times as many medications as those not recieing diabetes treatment[gw]. Another major driver of prescription use increase is the aging population; about 52 million Americans are of age 65 or older and taking the lions share of medications.

When considering this growth, it becomes more clear why medical/medication error would be identified as the third leading cause of death in the US in 2016. As patients take more medications and drugs become more potent and complex, the risk of interactions, side effects, and errors increase as well. In response to these errors that lead to heath deficiency or death, malpractice suits have grown to account for 5% of 


Carelessly prescribe the wrong medication
Give a patient the wrong dosage, unintentionally
Give a patient an unintended medication
Fail to take a history of the patient’s prescription drug use
Fail to warn of all risks associated with the medication
Dangerously mixed different prescription drugs

administering the wrong medication to a patient
administering the wrong dosage of medication (i.e., too much or too little medication)
mislabeling the medication
prescribing the patient a medication that the patient is allergic to
prescribing the patient a medication that interacts negatively with other medications that the patient is taking, and
failing to warn the patient of the common side effects of the medication.


The FDA provides guidance for defining, monitoring, and enforcing practices to reduce medical and medication errors. 

> **Medication Error:** Any preventable event that may cause or lead to inappropriate medication use or patient harm while the medication is in the control of the healthcare professional, patient, or consumer.
> 
> **Adverse Medical Event (ADE):** Any abnormal sign, symptom or laboratory test, or any syndromic combination of such abnormalities, any untoward or unplanned occurrence (e.g. an accident or unplanned pregnancy), or any unexpected deterioration in a concurrent illness
> 
> **Adverse Medical Reaction (ADR):** an appreciably harmful or unpleasant reaction, resulting from an intervention related to the use of a medicinal product. 

<p align="center">
   <img src=https://github.com/a-woodbury/RxID/blob/master/Images/ADE.png title="ADE venn" style="width:500px;height:600px;"/>
   <figcaption>Fig.1 - Medication Errors, Events, and Reactions.</figcaption>
</p>	</p>


When an ADE is identified, the course of action is clinical intervention with the patient and, if occurring in a clinical setting, reporting and correction action with the offending agency. However, as seen in the above graphic, not all ADEs are caused by medication error, and not all medication errors lead to an ADE or ADR. We will narrow our focus on ADEs  caused by medication error.

### Progress

In this project, we are concerned with medication safety from the perspective of a retail, mail-order, or inpatient pharmacy setting. RxID is certainly not the first pursuit of improvement in medication safety; the FDA has rolled out programs, regulations, and research to:

- Guide drug developers and manufacturers provide clearer reporting on drug ingredients, labeling, and manufacturing (not in scope of RxID);
- Help pharmacies reduce errors in dispensing and releasing medications;
- Assist patients in understanding the therapy and complications of their medicines.

Computerized prescription management, automated dispensing systems, and online prescription ordering have made it easier to get the right medication to the right patient quickly and safely.

#### Patient Challenges

Patients still struggle with identifying their medications, and often resort to inconsistent and risky behaviors, like putting all of their medications in one bottle or failing to comply with complicated therapy out of frustration. In the event that a patient needs assistance identifying a medication, they can check an online resource (risky), ask a caregiver, who may or may not be able to assist, or a pharmacist. Pharmacists are tasked with looking up a pill description, which may be fruitless or untimely, and this is a miss in service for the immediate patient and those patients who need the pharmacists time for clinical questions about their therapy (side effects, alternatives, dosing, etc). Many patients do not track their presciptions by the National Drug Code (NDC) and may need assistance validating if their medication was included in a recall, and pharmacies are often innundated with these questions during major recall events. 

##### Recommendation: Patient/Caregiver Support

A smartphone app to cross-reference an identified medication with a patients medication list would give them confidence and reduce the stress of managing their medications. Patients with visual impairment can be sure to distibguish between similar looking medications without needing to strain to identify imprints or scoring. Checking for the recall status of their medications will be effortless as well with this model readily available.

Ideally, the app implementation of this model would be localized to the smartphone; this way no Personal Health Information is transmitted and privacy can be assured.

#### Pharmacy Challenges

Pharmacies invest thousands and millions into their processes and techology stacks to ensure each order is filled accurately per the prescription and received by the designated patient or caregiver. Despite these efforts, medication errors still occur. Payroll hours have continued to drop due to increased competition and pharmacists find themselves taking on more administrative and retail tasks than they had previously, without the help of as many technicians. These pharmacists are under more stress to maintain safety and operate a profitable business, and this leads to fatigue and burnout, which leads to errors. A prescription released to the wrong patient or with the wrong drug is embarrassing, costly, and dangerous.

In regards to cost, it should be 

##### Recommendation: Dispensing Validation

To minimize  risk, implementing an image recognition model into the dispensing workflow can identify if a bottle has been filled incorrectly before the pharmacist does their visual inspection. Mail-order pharmacies already use cameras to take an overhead snapshot of the filled bottle for later validation if a patient is concerned, but using an image recognition model can proactively catch misfills before they could become errors. High-volume chain retail pharmacies would also benefit by taking advantage of the model to avoid sending orders back to technicians for filling correctly. 



#### Emergency Challenges

During medical emergencies, EMTs and nurses must work quickly to gather information about the patients symptoms, history, and medications. 
 
##### Recommendation: Emergency Intervention

If a collection of unsorted medications is found or provided, RxID could give insight into a patients condition, even if the medication is

### Pill Identification

We have identified projects and competitions challenging computer engineering teams and labs to create models and applications in pursuit of high-accuracy pill recognition. A major competition, and the spark to this project, was the National Library of Medicine's [Pill Image Recognition Challenge](https://pir.nlm.nih.gov/challenge/) in 2016. The winning team from Michigan State University produce [*MobileDeepPill*](https://www.egr.msu.edu/~mizhang/papers/2017_MobiSys_MobileDeepPill.pdf) [(github)](https://github.com/zhangmifigo/MobileDeepPill), which was able to correctly identify medications from the 4000+ image set with up to 97% accuracy in certain tests. The team prioritized efficiency and privacy in designing their Android App and published their research and code openly. Other teams and competitions did not bear fruit in the form of scores or literature, so it may still be a novel area of research in pharma to be pursued. With the wide range of scores and no major implementation in production for patient or clinic use, we suspect performance, cost, or regulations are preventing using image recognition models for this purpose.

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
