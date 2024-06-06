# Detect_Facial_Expressions
This project is  a real-time emotion recognition system that captures a user's facial expressions through a webcam, identifies the emotion, and displays the corresponding emoji. The system leverages deep learning techniques for emotion detection and integrates with a graphical user interface (GUI) for user interaction.

You can find the folder for the same in its branch named master.

#Usage
1) Setup: Ensure all required files (model weights, logo, emojis) are in their respective directories.
2) Run the Application: Execute the emoji.py script.
3) Real-time Emotion Detection: The application captures real-time video, detects emotions, and displays corresponding emojis and text labels.

FEATURES:
Model Architecture: The project uses a Convolutional Neural Network (CNN) to classify emotions from facial expressions. The model consists of multiple convolutional layers, max-pooling layers, dropout layers, and dense layers.
Training Data: The model is trained on grayscale images of faces, categorized into seven emotions: Angry, Disgusted, Fearful, Happy, Neutral, Sad, and Surprised.
Model Training: Training involves using the Keras library with TensorFlow backend, with data augmentation techniques to improve model generalization.

Real-time Capture: The system captures real-time video feed from the user's webcam using OpenCV.
Face Detection: Faces in the video stream are detected using Haar cascades, a pre-trained face detection method provided by OpenCV.

Framework: The GUI is built using Tkinter, the standard Python interface to the Tk GUI toolkit.
Display Elements: The interface includes a video feed window showing the real-time webcam input, an area displaying the predicted emotion as text, and another area showing the corresponding emoji image.
User Interaction: The GUI allows users to quit the application through a button.
