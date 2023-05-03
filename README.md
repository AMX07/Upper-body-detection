# Upper Body Detection - README

This repository contains the implementation of an upper body detection algorithm using the Haar Cascade Classifier in OpenCV. The primary purpose of this project is to detect the upper body of a person in images or real-time video streams, such as webcams or CCTV footage.

## Prerequisites

Before you can run the code, make sure you have the following dependencies installed:

1. Python 3.x
2. OpenCV (cv2) - Version 4.x
3. NumPy

You can install the required libraries using the following command:

```bash
pip install opencv-python numpy
```

## Usage

Clone the repository to your local machine using the following command:
```bash
git clone https://github.com/AMX07/Upper-body-detection.git
```

Navigate to the project folder:
```bash
cd Upper-body-detection
```

Run the all.py script:
```bash
python all.py
```

By default, the script will use your computer's webcam as the video source. If you want to use a different video source, you can modify the video_capture = cv2.VideoCapture(0) line in the all.py file, replacing 0 with the index of your desired video source.

To exit the program, press the q key while the video window is in focus.
