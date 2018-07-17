import os
import cv2
import numpy as np

# lbp_face = cv2.CascadeClassifier('opencv/data/lbpcascades/lbpcascade_frontalface.xml')
haar_face = cv2.CascadeClassifier('opencv/data/haarcascades/haarcascade_frontalface_alt.xml')
haar_eye = cv2.CascadeClassifier('opencv/data/haarcascades/haarcascade_eye.xml')
# haar_smile = cv2.CascadeClassifier('opencv/data/haarcascades/haarcascade_smile.xml')

data_folder_path = "images"

if not os.path.exists(data_folder_path):
    os.makedirs(data_folder_path)


def detect_faces(colored_img, f_cascade, scaleFactor=1.2):
    img_copy = np.copy(colored_img)
    # convert the test image to gray image as opencv face detector expects gray images
    gray = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)

    # let's detect multiscale (some images may be closer to camera than others) images
    faces = f_cascade.detectMultiScale(gray, scaleFactor=scaleFactor, minNeighbors=5)

    if len(faces) == 0:
        return img_copy, np.array([]), None

    # go over list of faces and draw them as rectangles on original colored img
    for face in faces:
        draw_rectangle(img_copy, face)

    (x, y, w, h) = faces[0]

    return img_copy, gray[y:y + w, x:x + h], faces[0]


def save_training_data(img, image_name):
    cv2.imwrite(os.path.join(data_folder_path, image_name+".jpg"), cv2.resize(img, (300, 300)))


def draw_rectangle(img, rect):
    (x, y, w, h) = rect
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)


def draw_text(img, text, x, y):
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)

