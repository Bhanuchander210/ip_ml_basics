# Code Used from PyImageSearch.
# Link : https://www.pyimagesearch.com/2017/04/03/facial-landmarks-dlib-opencv-python/

from imutils import face_utils
import imutils
import dlib
import cv2


file = '../data/avengers/thanos.jpeg'
detector = dlib.get_frontal_face_detector()

# Using the shape predictor file from dlib site
# Link http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2

predictor = dlib.shape_predictor("/home/bhanuchander/Downloads/shape_predictor_68_face_landmarks.dat")

image = cv2.imread(file)
image = imutils.resize(image, width=500)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# detect faces in the gray scale image
rects = detector(gray, 2)

print "Number of Faces Found : {}".format(len(rects))

for (i, rect) in enumerate(rects):
    # determine the facial landmarks for the face region, then
    # convert the facial landmark (x, y)-coordinates to a NumPy
    # array
    shape = predictor(gray, rect)
    shape = face_utils.shape_to_np(shape)

    # convert dlib's rectangle to a OpenCV-style bounding box
    # [i.e., (x, y, w, h)], then draw the face bounding box
    (x, y, w, h) = face_utils.rect_to_bb(rect)
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # show the face number
    cv2.putText(image, "Face #{}".format(i + 1), (x - 10, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # loop over the (x, y)-coordinates for the facial landmarks
    # and draw them on the image
    for (x, y) in shape:
        cv2.circle(image, (x, y), 1, (0, 0, 255), -1)

# show the output image with the face detections + facial landmarks
cv2.imshow("Output", image)
key = cv2.waitKey(5000)
if key == 27:
    cv2.destroyAllWindows()