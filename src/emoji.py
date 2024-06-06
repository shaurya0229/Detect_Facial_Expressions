import tkinter as tk
from tkinter import *
import cv2
from PIL import Image, ImageTk
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Input

model_path = 'C:/Users/shaur/OneDrive/Desktop/prject/src/emotion_model.weights.h5'
logo_path = 'C:/Users/shaur/OneDrive/Desktop/prject/src/data/logo.png'  # Update this if logo.png is located elsewhere
emoji_base_path = 'C:/Users/shaur/OneDrive/Desktop/prject/src/data/emojis/'

if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found: {model_path}")

if not os.path.exists(logo_path):
    raise FileNotFoundError(f"Logo file not found: {logo_path}")

required_emojis = ["angry.png", "disgusted.png", "fearful.png", "happy.png", "neutral.png", "sad.png", "surprised.png"]
for emoji_file in required_emojis:
    emoji_path = os.path.join(emoji_base_path, emoji_file)
    if not os.path.exists(emoji_path):
        raise FileNotFoundError(f"Emoji file not found: {emoji_path}")

emotion_model = Sequential()

emotion_model.add(Input(shape=(48, 48, 1)))
emotion_model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
emotion_model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Dropout(0.25))

emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Dropout(0.25))

emotion_model.add(Flatten())
emotion_model.add(Dense(1024, activation='relu'))
emotion_model.add(Dropout(0.5))
emotion_model.add(Dense(7, activation='softmax'))

emotion_model.load_weights(model_path)

cv2.ocl.setUseOpenCL(False)

emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}
emoji_dist = {
    0: os.path.join(emoji_base_path, "angry.png"),
    1: os.path.join(emoji_base_path, "disgusted.png"),
    2: os.path.join(emoji_base_path, "fearful.png"),
    3: os.path.join(emoji_base_path, "happy.png"),
    4: os.path.join(emoji_base_path, "neutral.png"),
    5: os.path.join(emoji_base_path, "sad.png"),
    6: os.path.join(emoji_base_path, "surprised.png")
}

global last_frame1
last_frame1 = np.zeros((480, 640, 3), dtype=np.uint8)
global cap1
show_text = [0]


def show_vid():
    video_cap = cv2.VideoCapture(0)
    if not video_cap.isOpened():
        print("Cannot open the camera")
        return
    flag1, frame1 = video_cap.read()

    if not flag1:
        print("Failed to grab frame from webcam")
        lmain.after(10, show_vid)
        return

    frame1 = cv2.resize(frame1, (600, 500))

    bounding_box = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray_frame = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    num_faces = bounding_box.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in num_faces:
        cv2.rectangle(frame1, (x, y - 50), (x + w, y + h + 10), (255, 0, 0), 2)
        roi_gray_frame = gray_frame[y:y + h, x:x + w]
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)
        prediction = emotion_model.predict(cropped_img)
        maxindex = int(np.argmax(prediction))
        show_text[0] = maxindex

    global last_frame1
    last_frame1 = frame1.copy()
    pic = cv2.cvtColor(last_frame1, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(pic)
    imgtk = ImageTk.PhotoImage(image=img)
    lmain.imgtk = imgtk
    lmain.configure(image=imgtk)
    lmain.after(10, show_vid)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        exit()


def show_vid2():
    frame2 = cv2.imread(emoji_dist[show_text[0]])
    if frame2 is None:
        print(f"Error: Unable to read image file {emoji_dist[show_text[0]]}")
        return
    pic2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
    img2 = Image.fromarray(pic2)
    imgtk2 = ImageTk.PhotoImage(image=img2)
    lmain2.imgtk2 = imgtk2
    lmain3.configure(text=emotion_dict[show_text[0]], font=('arial', 45, 'bold'))
    lmain2.configure(image=imgtk2)
    lmain2.after(10, show_vid2)


if __name__ == '__main__':
    root = tk.Tk()
    img = ImageTk.PhotoImage(Image.open(logo_path))
    heading = Label(root, image=img, bg='black')
    heading.pack()

    heading2 = Label(root, text="Photo to Emoji", pady=20, font=('arial', 45, 'bold'), bg='black', fg='#CDCDCD')
    heading2.pack()

    lmain = tk.Label(master=root, padx=50, bd=10)
    lmain2 = tk.Label(master=root, bd=10)
    lmain3 = tk.Label(master=root, bd=10, fg="#CDCDCD", bg='black')
    lmain.pack(side=LEFT)
    lmain.place(x=50, y=250)
    lmain3.pack()
    lmain3.place(x=960, y=250)
    lmain2.pack(side=RIGHT)
    lmain2.place(x=900, y=350)

    root.title("Photo To Emoji")
    root.geometry("1400x900+100+10")
    root['bg'] = 'black'
    exitbutton = Button(root, text='Quit', fg="red", command=root.destroy, font=('arial', 25, 'bold')).pack(side=BOTTOM)

    show_vid()
    show_vid2()
    root.mainloop()
