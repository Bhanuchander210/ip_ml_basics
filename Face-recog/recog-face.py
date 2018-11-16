import face_recognition
import cv2
import numpy as np

from imutils import paths

dir = "../data/avengers/"

for file in paths.list_images(dir):

    cvimg = cv2.imread(file)
    # Resizing some times fails to recognize the face.
    # cvimg = cv2.resize(cvimg, (512, 512))

    face_locations = face_recognition.face_locations(cvimg)
    print("Face Recogntion Library found {} face(s) in this photograph.".format(len(face_locations)))

    def cvimshow(title, img):
        cv2.imshow(title, img)
        key = cv2.waitKey(5000)
        if key == 27:
            cv2.destroyAllWindows()

    cvimshow('show', cvimg)

    for i, face_location in enumerate(face_locations):
        top, right, bottom, left = face_location
        cv2.rectangle(cvimg, (left, top), (right, bottom), (0, 255, 0), 3)
        cv2.putText(cvimg, "Face #{}".format(i + 1), (left - 10, top - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cvimshow('Faces', cvimg)

    face_landmarks_list = face_recognition.face_landmarks(cvimg)
    for face_landmarks in face_landmarks_list:
        for facial_feature in face_landmarks.keys():
            lines = np.array(face_landmarks[facial_feature], np.int32)
            cv2.polylines(cvimg, [lines], True, (0, 255, 255))

    cvimshow('Facial Land Marks', cvimg)