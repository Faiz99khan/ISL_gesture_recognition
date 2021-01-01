# ISL_hand_gesture_recognition_in_real-time

## Table of Contents
  * [Results](#results)
  * [Overview](#overview)
  * [Abstract](#abstract)
  * [Installation](#installation)
  * [Run](#run)
  * [Training](#training)
  * [Pretrained models](#pretrained-models)
  * [Technologies Used](#technologies-used)
  * [License](#license)
  * [Credits](#credits)

## Results

## Overview
It is a vision-based system in which deep 3d CNN arhitecture is used to recognize ISL hand gesture in real-time and video using tranfer learning. It recognize 10 ISL hand gestures for numeric digits (0-9) in which all are static gesture except gesture for 6 which is dynamic. But, it can be extended for large no. of gesture classes without requiring huge amount of data. It gives around 85 % accuracy on video stream.  

## Abstract
Real-time recognition of ISL hand gestures using vision-based system is a challenging task because there is no indication when a dynamic gesture is starts and ends in a video stream and there is no ISL data publically available unlike ASL, to work on. In this work, we handle these challenges by doing transfer learning and operating deep 3D CNN architecture using sliding window approach. Sliding window approach suffers with multiple time activations problem but we remove it by doing some post processing. To find the region of interest(RoI) is also a difficult task, we solve this using face detection algorithm. The proposed architecture consists of two models: (1) A detector which is a lightweight CNN architecture to detect gestures and (2) a classifier which is a deep CNN to classify the detected gestures. To measure misclassiﬁcations, multiple detections, and missing detections at the same time, we use Levenshtein Distance. Using this, we find Levenshtein accuracy on video stream. We create our own data set of 10 ISL hand gestures for numeric digits(0-9), in which just 70 samples are created for each gesture class. We fine tune ResNeXt-101 model on our data set, which is used as a classifier, achieves good classification accuracy of 95.79 % and 94.39 % on training set and validation set respectively and around 85 % considerable accuracy on video stream.

## Installation
Just install the necessary libraries mentioned in the [requirements.txt](requirements.txt).

## Run
To run the app, just run this command after cloning the repository, installing the necessary libraries and downloading the models.
```bash
python app.py
```
Note: I tested it only on windows not on other os platforms like Linux, macOS.

## Training
I used Google colab GPU to train or fine tune the classifier.

Use [training.ipynb]() to train or fine tune the classifier on Google colab GPU or your own GPU

## Pretrained models
Download pretrained ResNeXt_101 classifier model from [here](https://drive.google.com/uc?export=download&id=1W-jNAvfjSwXghmiFTNu7hHEFFS5pEgEJ), which is trained on [jester](https://20bn.com/datasets/jester) largest dynamic hand gesture dataset.

Download pretrained ResNetl_10 detector model from [here](https://drive.google.com/uc?export=download&id=19rQQUKuzqjX2V0K9xIpvcjs1Fdi-1UjB), which is trained on [Egogesture](http://www.nlpr.ia.ac.cn/iva/yfzhang/datasets/egogesture.html) hand gesture dataset. 

Download fine tuned ResNeXt_101 classifier model from [here](https://drive.google.com/uc?export=download&id=1W-jNAvfjSwXghmiFTNu7hHEFFS5pEgEJ), which is fine tuned on our ISL hand gesture dataset.

Note: To run the app you would just need detector and classifier, after downloading, place them in same directory where all other files are present.

## Technologies used
![](https://forthebadge.com/images/badges/made-with-python.svg)

[<img target="_blank" src="https://upload.wikimedia.org/wikipedia/commons/9/96/Pytorch_logo.png" width=350 >](https://pytorch.org/) [<img target="_blank" src="https://upload.wikimedia.org/wikipedia/commons/5/53/OpenCV_Logo_with_text.png" width=150 >](https://docs.opencv.org/master/d0/de3/tutorial_py_intro.html) &nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp; [<img target="_blank" src="https://raw.githubusercontent.com/python-pillow/pillow-logo/master/pillow-logo-248x250.png" width=150>](https://pillow.readthedocs.io/en/stable/installation.html) 

[<img target="_blank" src="https://matplotlib.org/3.2.1/_static/logo2_compressed.svg" width=300>](https://matplotlib.org/) &nbsp;&nbsp;&nbsp;&nbsp; [<img target="_blank" src="https://flask.palletsprojects.com/en/1.1.x/_images/flask-logo.png" width=200>](https://flask.palletsprojects.com/en/1.1.x/)

## License
Licensed under [MIT Licencse](LICENSE)

## Credits
I thank Okan Köpüklü, Ahmet Gündüz et al. for providing the [codebase](https://github.com/ahmetgunduz/Real-time-GesRec) and i build this project on top of that.

I also thank my freinds Kunal Singh Bhandari, Mohd. Bilal and Digant Bhanwariya who all helped me in Web App design and data creation.

I also want to thank to Google for providing [free Colab GPU](https://colab.research.google.com/notebooks/intro.ipynb) service to everyone, due to which i was able to train the model.
