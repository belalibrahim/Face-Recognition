import os
import cv2
import numpy as np

lbp_face = cv2.CascadeClassifier('opencv/data/lbpcascades/lbpcascade_frontalface.xml')
haar_face = cv2.CascadeClassifier('opencv/data/haarcascades/haarcascade_frontalface_alt.xml')
haar_eye = cv2.CascadeClassifier('opencv/data/haarcascades/haarcascade_eye.xml')
# haar_smile = cv2.CascadeClassifier('opencv/data/haarcascades/haarcascade_smile.xml')

face_recognizer = cv2.face.EigenFaceRecognizer_create()


def detect_faces(f_cascade, colored_img, scaleFactor=1.2):
    img_copy = np.copy(colored_img)
    # convert the test image to gray image as opencv face detector expects gray images
    gray = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)

    # let's detect multiscale (some images may be closer to camera than others) images
    faces = f_cascade.detectMultiScale(gray, scaleFactor=scaleFactor, minNeighbors=5)

    if len(faces) == 0:
        return img_copy, np.array([]), None

    # go over list of faces and draw them as rectangles on original colored img
    for (x, y, w, h) in faces:
        cv2.rectangle(img_copy, (x, y), (x + w, y + h), (0, 255, 0), 2)

    (x, y, w, h) = faces[0]

    return img_copy, gray[y:y + w, x:x + h], faces[0]


cam = cv2.VideoCapture(0)

while cam.isOpened():

    _, frame = cam.read()
    frame = cv2.flip(frame, 1)

    frame, face, face_area = detect_faces(lbp_face, frame)

    cv2.imshow('Camera', frame)
    if list(face):
        cv2.imshow('Face', face)

    k = cv2.waitKey(10)
    if k == 27:
        break

    elif k == ord('c'):
        pass

    elif k == ord('t'):
        pass

    elif k == ord('r'):
        pass
