from Utils import *


cam = cv2.VideoCapture(0)

while cam.isOpened():

    _, frame = cam.read()
    frame = cv2.flip(frame, 1)

    frame, face, face_area = detect_faces(frame, haar_face)

    cv2.imshow('Camera', frame)

    k = cv2.waitKey(10)
    if k == 27:
        break

    elif k == ord('c'):
        image_name = input("Enter the image name:\n")
        save_training_data(face, image_name)
        print("Image captured successfully!")

    elif k == ord('t'):
        pass

    elif k == ord('p'):
        pass
