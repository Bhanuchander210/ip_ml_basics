import imutils
import time
import numpy as np
import cv2
import imutils

print 'Gotcha..'

CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"]

IGNORE = set(["chair"])

def cvimshow(title, img):
    cv2.imshow(title, img)
    key = cv2.waitKey(2000)
    if key == 27:
        cv2.destroyAllWindows()

COLORS = np.random.uniform (0, 255, size=(len(CLASSES), 3))

# load our serialized model from disk
print("[INFO] loading model...")

mobssdHome = "/home/bhanuchander/mlclones/MobileNet-SSD"

prot = mobssdHome + "/template/MobileNetSSD_deploy_template.prototxt"
model = mobssdHome + "/mobilenet_iter_73000.caffemodel"

print prot
print model

frame = cv2.imread("/home/bhanuchander/Pictures/mobssd/cars.jpeg", cv2.IMREAD_GRAYSCALE)

cvimshow("test", frame)

frame = imutils.resize(frame, width=400)

net = cv2.dnn.readNetFromCaffe(prot, model)

# grab the frame dimensions and convert it to a blob
(h, w) = frame.shape[:2]
blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
                             0.007843, (300, 300), 127.5)

# pass the blob through the network and obtain the detections and
# predictions
net.setInput(blob)
detections = net.forward()

# loop over the detections
for i in np.arange(0, detections.shape[2]):
    # extract the confidence (i.e., probability) associated with
    # the prediction
    confidence = detections[0, 0, i, 2]

    # filter out weak detections by ensuring the `confidence` is
    # greater than the minimum confidence
    if confidence > args["confidence"]:
        # extract the index of the class label from the
        # `detections`
        idx = int(detections[0, 0, i, 1])

        # if the predicted class label is in the set of classes
        # we want to ignore then skip the detection
        if CLASSES[idx] in IGNORE:
            continue
    # compute the (x, y)-coordinates of the bounding box for
    # the object
    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
    (startX, startY, endX, endY) = box.astype("int")

    # draw the prediction on the frame
    label = "{}: {:.2f}%".format(CLASSES[idx],
                                 confidence * 100)
    cv2.rectangle(frame, (startX, startY), (endX, endY),
                  COLORS[idx], 2)
    y = startY - 15 if startY - 15 > 15 else startY + 15
    cv2.putText(frame, label, (startX, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)

cvimshow("FRAME", frame)