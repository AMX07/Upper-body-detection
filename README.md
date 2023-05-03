# Upper-body-detection

Upper Body Detection - README
This repository contains the implementation of an upper body detection algorithm using the Haar Cascade Classifier in OpenCV. The primary purpose of this project is to detect the upper body of a person in images or real-time video streams, such as webcams or CCTV footage.

Prerequisites
Before you can run the code, make sure you have the following dependencies installed:

Python 3.x
OpenCV (cv2) - Version 4.x
NumPy
You can install the required libraries using the following command:

bash
Copy code
pip install opencv-python numpy
Usage
Clone the repository to your local machine using the following command:
bash
Copy code
git clone https://github.com/AMX07/Upper-body-detection.git
Navigate to the project folder:
bash
Copy code
cd Upper-body-detection
Run the all.py script:
bash
Copy code
python all.py
By default, the script will use your computer's webcam as the video source. If you want to use a different video source, you can modify the video_capture = cv2.VideoCapture(0) line in the all.py file, replacing 0 with the index of your desired video source.

To exit the program, press the q key while the video window is in focus.
How It Works
The script leverages the Haar Cascade Classifier, a machine learning object detection method, to detect the upper body of a person in the video stream. The classifier is trained on positive and negative images of upper body parts, and it uses this knowledge to identify the upper body in the input video.

The repository includes a pre-trained Haar Cascade file for upper body detection (haarcascade_upperbody.xml). This file is loaded by the script, and the classifier is applied to the video frames to detect and draw bounding boxes around the upper body of people in the video stream.

Contributing
We welcome contributions to improve the performance and accuracy of the upper body detection algorithm. Feel free to submit issues, feature requests, or pull requests.
