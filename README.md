# Face Recognition System

A C++ face detection and face recognition project built with OpenCV.

## Features

- Detects faces using YuNet
- Recognizes faces using SFace
- Uses OpenCV for image processing
- Supports real-time face detection from a camera
- Compares detected faces against saved face data

## Technologies Used

- C++
- OpenCV
- YuNet face detection model
- SFace face recognition model
- VS Code

## Project Files

- `main.cpp` — main program
- `face_detection_yunet_2023mar.onnx` — YuNet face detection model
- `face_recognition_sface_2021dec.onnx` — SFace face recognition model
- `haarcascade_frontalface_default.xml` — Haar Cascade face detector
- `.vscode/` — VS Code project settings

## How It Works

The program:

1. Captures an image or video frame
2. Detects faces in the frame
3. Extracts facial features
4. Compares those features with known faces
5. Determines whether the detected face matches a saved person

## Requirements

You need:

- C++ compiler
- OpenCV
- A webcam or camera
- The included ONNX model files

## Privacy

This repository does not include personal face images or private biometric data.

## Future Improvements

Possible future upgrades:

- Improve recognition accuracy
- Add support for multiple known users
- Add a graphical interface
- Connect the system to a voice assistant
- Improve real-time performance