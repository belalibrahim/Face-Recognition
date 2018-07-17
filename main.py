from Utils import *


cam = cv2.VideoCapture(0)

print("Training data...")
faces, labels = prepare_training_data()

print("Total faces: ", len(faces))
print("Total labels: ", len(labels))

if len(faces) > 0:
    face_recognizer = cv2.face.EigenFaceRecognizer_create()
    face_recognizer.train(faces, np.array(labels))
else:
    print("There is no image to train.")

while cam.isOpened():

    _, frame = cam.read()
    frame = cv2.flip(frame, 1)

    frame, face, face_area = detect_faces(frame, haar_face)

    draw_text(frame, "Press 'c' to capture a new image", 20, 40)
    draw_text(frame, "Press 't' to train the captured images", 20, 70)
    draw_text(frame, "Press 'p' to predict a new image", 20, 100)

    cv2.imshow('Camera', frame)

    k = cv2.waitKey(10)
    if k == 27:
        break

    elif k == ord('c'):
        if face is not None:
            image_name = input("Enter the image name:\n")
            save_training_data(face, image_name)
            print("Image captured successfully!")
        else:
            print("Unrecognized face!")

    elif k == ord('t'):
        faces, labels = prepare_training_data()
        print("Total faces: ", len(faces))
        print("Total labels: ", len(labels))
        if len(faces) > 0:
            print("Training data...")
            face_recognizer = cv2.face.EigenFaceRecognizer_create()
            face_recognizer.train(faces, np.array(labels))
            print("Data trained successfully!")
        else:
            print("There is no image to train.")

    elif k == ord('p'):
        pass
