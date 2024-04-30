# ASL Sign Language Detector

This is a brief documentation about the sign language detector tech for the game - [Gesture Quest]().  

[Gesture Quest]() is an educational game designed to educate individuals on ASL (American Sign Language). 

## Install
* Python 3.11
* pip install opencv-python
* pip install mediapipe
* pip install numpy
* pip install tensorflow

## Directory
  
This repo contains three modules:  
* Collector: This is the data collector module for extracting data from hand landmarks provided by [Google MediaPipe](https://developers.google.com/mediapipe). 
* Reader: This is the simple app reader module to showcase the trained ML model prediction from the video/webcam live stream.
* Trainer: This is the classifier model training notebook to train both static signs and movement signs.

<pre>
├─ Collector
│   |
│   ├─  collect.py
│   ├─  data.csv
│   ├─  label.csv
│   └─  m_data.csv
|
├─ Reader
│   |
|   ├─  model
│   │    |
│   │    ├─  classifier.py
│   |    ├─  label.csv
│   │    ├─  m_model.tflite
│   |    └─  model.tflite
│   |
|   └─  app.py
|
└─ Trainer
    |
    ├─  model
    │    |
    │    ├─  data
    │    |    |
    │    |    ├─  data.csv
    |    |    ├─  label.csv
    |    |    └─  m_data.csv
    |    |
    |    └─   ## trained models
    |
    ├─  m_model_trainer.ipynb
    └─  model_trainer.ipynb
</pre>

### app.py
Execution file. 
Loads mediapipe model and classifier for detection.
Calls classifier and maps detection results with labels.

### *Data*:
* **landmark_list** is the 21 landmarks coordinates, with only x and y coordinates included. 
* **landmark_list_3D** is the 21 landmarks coordinates, with x, y and z coordinates included.
* **which_hand** outputs a string for which hand is detected.
* **N_hand_sign_id** is the index number of th top N result labels.
* **N_hand_sign_probs** is the probability of the top N result labels.

* #### In the for loop:
    * **N_detected_elements** is a pair containing the **INDEX** and **PROBABILITY** of the detected gesture

    * **N_detected_elements[0]** is the **INDEX** of the detected gesture

    * **N_detected_elements[1]** is the **PROBABILITY** of the detected gesture


### classifier.py
Loads tflite model, reads landmarks through model and returns top N confidence score indices and probabilities of the recognition output.

* Change the **MAX_N** for the top N results of the classifier output.


### label.csv
Use for read label with id from classifier output.

### m_model.tflite
Movement sign model file, replacable.

### model.tflite
Static sign model file, replacable.

# Reference
* [MediaPipe](https://mediapipe.dev/)
* [Kazuhito Takahashi](https://twitter.com/KzhtTkhs)
* [Nikita Kiselov](https://github.com/kinivi)

# Author
* [CMU-ETC-Questure](https://projects.etc.cmu.edu/questure/)
* [Tyler Yang](https://github.com/ZWeiweiY)


# License
* Collector, Reader, Trainer is under [Apache v2 license](LICENSE)
