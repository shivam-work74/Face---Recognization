# Face---Recognization
# Face Recognition Project 👤

This is a simple face recognition project I built using **Python** and **OpenCV**.  
The goal was to understand how face detection and recognition actually work, and to create a small working demo that can recognize people using my webcam.  

---

## How it Works
1. **Enroll Faces** → Run `enroll_faces.py` to capture around 50 images of a person. The images are saved inside `dataset/PersonName`.  
2. **Train Model** → Run `trainer.py` which trains the recognizer on all the images inside the dataset and saves the model in the `trainer/` folder.  
3. **Recognize Faces** → Run `recognizer.py` and the webcam will try to recognize the person in front of the camera. If it finds a match, it shows the name and confidence score, otherwise it shows "Unknown".  

---

## Installation & Setup
Clone this repo and install the dependencies:
```bash
git clone https://github.com/<your-username>/Face---Recognization.git
cd Face---Recognization
pip install opencv-python opencv-contrib-python numpy pillow
