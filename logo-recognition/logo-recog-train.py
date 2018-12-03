# import the necessary packages
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from skimage import exposure
from skimage import feature
from imutils import paths
import argparse
import imutils
import cv2
from sklearn.metrics import accuracy_score
from sklearn.externals import joblib

# construct the argument parse and parse command line arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--training", required=True, help="Path to the logos training dataset")
args = vars(ap.parse_args())

# initialize the data matrix and labels
print "[INFO] extracting features..."
data = []
labels = []


def cvimshow(title, img):
    cv2.imshow(title, img)
    key = cv2.waitKey(2000)
    if key == 27:
        cv2.destroyAllWindows()

# loop over the image paths in the training set
for imagePath in paths.list_images(args["training"]):

    try:
    # extract the make of the car
        make = imagePath.split("/")[-2]

        # load the image, convert it to grayscale, and detect edges

        image = cv2.imread(imagePath)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edged = imutils.auto_canny(gray)

        # find contours in the edge map, keeping only the largest one which
        # is presmumed to be the car logo
        cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if imutils.is_cv2() else cnts[1]
        if cnts == []:
            continue
        c = max(cnts, key=cv2.contourArea)

        # extract the logo of the car and resize it to a canonical width
        # and height
        (x, y, w, h) = cv2.boundingRect(c)
        logo = gray[y:y + h, x:x + w]
        logo = cv2.resize(logo, (256, 256))

        # extract Histogram of Oriented Gradients from the logo
        H = feature.hog(logo, orientations=9, pixels_per_cell=(10, 10),
                        cells_per_block=(2, 2), transform_sqrt=True, block_norm="L2-Hys")

        # update the data and labels
        data.append(H)
        labels.append(make)
    except:
        print "cant load file : "+ imagePath


# "train" the nearest neighbors classifier
print 'Total Classes : ', set(labels)
print("[INFO] Appending Classifier Modules")

models = []
# models.append(('LR', LogisticRegression()))
# models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier(n_neighbors=1)))
# models.append(('CART', DecisionTreeClassifier()))
# models.append(('NB', GaussianNB()))
# models.append(('SVM', SVC()))

for name, model in models:

    model.fit(data, labels)
    pred = model.predict(data)
    print 'Module Name : ', name
    print 'Training Accuracy Score : ', accuracy_score(labels, pred)
    filename = name + '_model.sav'
    joblib.dump(model, filename)

    # This part has been changed as python Test. This code only do train.
    # print("[INFO] evaluating...")
    #
    # # loop over the test dataset
    # for (i, imagePath) in enumerate(paths.list_images(args["test"])):
    #     # load the test image, convert it to grayscale, and resize it to
    #     # the canonical size
    #     image = cv2.imread(imagePath)
    #
    #     gray = cv2.cvtColor (image, cv2.COLOR_BGR2GRAY)
    #     logo = cv2.resize (gray, (256, 256))
    #
    #     # extract Histogram of Oriented Gradients from the test image and
    #     # predict the make of the car
    #     (H, hogImage) = feature.hog(logo, orientations=9, pixels_per_cell=(10, 10),
    #                                 cells_per_block=(2, 2), transform_sqrt=True, block_norm="L2-Hys", visualise=True)
    #     pred_proba = model.predict_proba(H.reshape(1, -1))
    #     pred = model.predict(H.reshape(1, -1))
    #     print imagePath , ' : ', pred
    #
    #     # visualize the HOG image
    #     hogImage = exposure.rescale_intensity(hogImage, out_range=(0, 255))
    #     hogImage = hogImage.astype("uint8")
    #     cvimshow("HOG Image #{}".format(i + 1), hogImage)
    #
    #     # draw the prediction on the test image and display it
    #     cv2.putText(image, str(pred), (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 3)
    #     cvimshow("Test Image #{}".format(i + 1), image)
