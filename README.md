# organelle_classifier
Classify organelles from images of microscopic titration of several cells.


## Table of content

* [0. Modules and versions](#modules)
* [1. About the context of this study](#content)
* [2. About the HPA-SCC competition](#competition)
  * [a. Context](#context)
  * [b. Links](#links)
  * [c. Scoring](#scoring)
* [3. Models used](#models)
  * [a. HPA Cell Segmentator](#segmentator)
  * [b. Organelle Classifier](#org_class)
* [4. API](#api)
  * [a. Installation](#install)
  * [b. How to use](#howto)
* [5. Thanks](#thanks)
	
## 0. Modules and versions <a name="modules">
	
* click 7.0
* Flask 2.0.1
* imageio 2.9.0
* importlib-metadata 3.10.0
* numpy 1.19.5
* pillow 8.2.0
* pydantic 1.8.2
* opencv-python 4.5.1.48
* scikit-image 0.16.2
* scipy 1.4.1
* tensorflow 2.4.1
* torch 1.4.0
* torchvision 0.5.0

## 1. About the context of this study <a name="content">

This study is part of our training in Deep Learning and Image Recognition.

As our final projet, we chose to study the dataset Human Protein Atlas - Single Cell Classification.
It was a challenging project. What led us to work on it was its apperture to new ideas and new ways to handle several parts of the necessary process to get a correct prediction.

## 2. About the Human Protein Atlas - Single Cell Classification competition <a name="competition">


### a. Context  <a name="context">
  
![Kaggle HPA - Single Cell Classification](/img/hpa_kaggle.PNG)

The goal is to design an algorithm able to detect the organelle highlighted by the green channel of an image.
  
Each image is composed of four channels. The naming scheme of those channels is arbitrary as they are not to be associated with true colors : they just represent different microscopical observation of the same view.
Red :  microtubule
Blue : nuclei
Yellow : endoplasmic reticulum (ER) channels
Green for protein of interest of which we try to predict the class.

Those organelle are our outputs and are classified between 18 classes :
- Nucleoplasm
- Nuclear membrane
- Nucleoli
- Nucleoli fibrillar center
- Nuclear speckles
- Nuclear bodies
- Endoplasmic reticulum
- Golgi apparatus
- Intermediate filaments
- Actin filaments
- Microtubules
- Mitotic spindle
- Centrosome
- Plasma membrane
- Mitochondria
- Aggresome
- Vesicles and punctate cytosolic patterns
... and one Negative class.


### b. Links  <a name="links">

Human Protein Atlas - Single Cell Classification competition : https://www.kaggle.com/c/hpa-single-cell-image-classification

### c. Loss and scoring  <a name="scoring">

The score used for this project was the categorical cross-entropy loss. It's a metric that needs to be minimized.

Here is its formula, for C classes, S samples and p meaning "positive" :

![Categorical cross entropy](/img/categorical_cross_entropy.jpg)
	
The evaluation metrics is the classic categorical accuracy.

## 3. Models used <a name="models">
  
### a. HPA Cell Segmentator  <a name="segmentator">

A Deep learning model based on the infamous "U-NET" but greatly improved and enriched by the original team.
  
![Original U-Net](/img/unet.png)
  
Its purpose is to locate and instanciate each individual cell in an image composed of the three aformentioned channels (except the "green").
  
You can find more information about this model here : https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/
The GitHub of the HPA_Cell_Segmentator is accessible here : https://github.com/CellProfiling/HPA-Cell-Segmentation
  
### b. Organelle Classifier  <a name="org_class">
  
A custom Tensorflow model currently in development.
  
Please read the R&D notebooks "Building and tuning the model" to get some details about its architecture, available in this repertory /https://github.com/WGribaa/organelle_classifier/tree/main/R%26D%20Notebooks%20and%20presentation .
	
Please read the pdf file "Cell_level_segmentation - presentation (FR).pd" (available in the same repertory) to read about our intuitions and what we achieved sor far.
  
## 4. API <a name="api">

### a. Installation  <a name="install">
  
Easy to use, you can simple launch "app.py" and let the script install the cnessary HPA_Cell_Segmentator from GitHub. Just ensure all the modules in requirements.txt are installed in your environment.
Once launched, you should acces the API Flask interface in any Web Browser at the address: "localhost:5000*".

### a. How to use  <a name="howto">
  
![API Home page](/img/api_home.PNG)
  
Click on "Parcourir" for each channel and make sure you send :
  - image according to the correct channels.
  - four images which belong to the same observation/sample.
  
Then please wait. Dependiong on your hardware, the HPA Cell Segmentator and our Tensorflow "Organelle Classifier" could take up to 30 seconds.
  
Once the loading and predicting process are over, the result will appear in another page.
  
![API Result page](/img/api_result.PNG)
 
The recomposed nucleoli, microtubule and endoplasmic reticulum channels will appear, recombined, in the top-left image.
A view of the mask and the "green" channel of interest, as well as the cell-wise bounding boxes, will appear in the top-right image.

Your four sent images will appeared, colored in the middle.
  
The prediction and confidence (namely "Prédiction" and "Confiance") will be explicitely displayed in the bottom.
  
You can then click "Accueil" to make a new prediction.

  
## 5. Thanks  <a name="thanks">

Thanks to our Image Recognition teacher, Pedro Octaviano, who spent inconsiderate amount of his free time to teach and help us produce the best we could.


 * Made by: Wael GRIBAA, Amine OMRI, Sabina REXHA, Abdou Akim GOUMBALA and Meryem GASSAB
 * Date: 25th May 2021
