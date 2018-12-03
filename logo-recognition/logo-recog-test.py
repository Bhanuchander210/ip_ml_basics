import imutils
from skimage import exposure
from skimage import feature
from imutils import paths
import cv2
import argparse
from sklearn.externals import joblib

def cvimshow(title, img):
    cv2.imshow(title, img)
    key = cv2.waitKey(2000)
    if key == 27:
        cv2.destroyAllWindows()

# load the model from disk
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--module", required=True, help="Path to pre-trained module")
ap.add_argument("-t", "--test", required=True, help="Path to the test dataset")
args = vars(ap.parse_args())
loaded_model = joblib.load(args['module'])

print("[INFO] evaluating...")

# loop over the test dataset
for (i, imagePath) in enumerate(paths.list_images(args["test"])):
    # load the test image, convert it to grayscale, and resize it to
    # the canonical size
    image = cv2.imread(imagePath)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    logo = cv2.resize(gray, (256, 256))

    # extract Histogram of Oriented Gradients from the test image and
    # predict the make of the car
    (H, hogImage) = feature.hog(logo, orientations=9, pixels_per_cell=(10, 10),
                                cells_per_block=(2, 2), transform_sqrt=True, block_norm="L2-Hys", visualise=True)
    pred_proba = loaded_model.predict_proba(H.reshape(1, -1))
    pred = loaded_model.predict(H.reshape(1, -1))
    print imagePath, ' : ', pred

    print '----------------------'
    print 'Histo : ', pred_proba
    print '----------------------'

    # visualize the HOG image
    hogImage = exposure.rescale_intensity(hogImage, out_range=(0, 255))
    hogImage = hogImage.astype("uint8")
    cvimshow("HOG Image #{}".format(i + 1), hogImage)

    # draw the prediction on the test image and display it
    cv2.putText(image, str(pred), (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 3)
    cvimshow("Test Image #{}".format(i + 1), image)