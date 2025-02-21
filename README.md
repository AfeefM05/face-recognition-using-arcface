# face-recognition-using-arcface
This project implements real-time face recognition using SCRFD 2.5G for face detection and ArcFace ResNet 100 for face feature extraction and recognition. The project also leverages CuPy for GPU-accelerated computing and threading for improved performance.

# Folder Structure
```
├── dataset
│   └── new_persons
│       └── <person1_name>
│       └── <person2_name>
│       └── ....
│   └── data
│   └── backup
    └── facial_features.npz
```

# Usage
- # Adding a New Person
To add a new person to the dataset:

Create a folder named after the person (e.g., John_Doe) inside dataset/new_persons/.

Place images of the person in the folder. The images should ideally be varied (different angles, lighting, etc.) for better recognition.

Run add_person.py to extract the facial features and store them as .npz files.

- # Real-Time Face Recognition
To run real-time face recognition:

Ensure the dataset folder contains the feature files for the known individuals.

Run the recognize.py script to:

- Capture video frames from your webcam (or any video source).
- Detect faces in real-time using SCRFD.
- Extract facial features using ArcFace ResNet 100.
- Compare the extracted features with the stored features from the dataset to recognize individuals.
- The recognized faces will be displayed with labels corresponding to the person's name.

# How It Works
- Face Detection (SCRFD 2.5G): The SCRFD (Single-stage Face Re-Detection) algorithm is used for detecting faces in images or video streams. It is fast and highly accurate.

- Face Feature Extraction (ArcFace ResNet 100): ArcFace is used to extract deep learning-based facial features that are robust to pose, expression, and lighting variations. The ResNet 100 model is used to encode the face into a high-dimensional feature vector.

- Real-time Recognition: With threading and GPU acceleration via CuPy, the system runs efficiently for real-time face recognition, ensuring smooth and fast performance.

- Data Storage: The extracted features for each individual are stored in .npz files inside the dataset/ directory. These feature vectors are then used for recognition in recognize.py.
