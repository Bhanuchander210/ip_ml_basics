# This code used to read the image and separate the color regions.
# I have used this to find the network / Tree counts in a network/graph image.

import numpy as np
from skimage import io, morphology, measure
from sklearn.cluster import KMeans
import cv2

def cvimshow(title, img):
    cv2.imshow(title, img)
    key = cv2.waitKey(20000)
    if key == 27:
        cv2.destroyAllWindows()

import glob
for imgpath in glob.glob("../data/network/*.png"):
    img = io.imread(imgpath)

    img[img < 255] = 0
    rows, cols, bands = img.shape
    X = img.reshape(rows*cols, bands)

    kmeans = KMeans(n_clusters=5, random_state=0).fit(X)
    labels = kmeans.labels_.reshape(rows, cols)

    # As we know that,
    classes = {0: 'Networks', 510: 'Nodes'}

    line = 1
    for i in np.unique(labels):
        blobs = np.int_(morphology.binary_opening(labels == i))
        color = np.around(kmeans.cluster_centers_[i])
        count = len(np.unique(measure.label(blobs))) - 1
        if (color.sum() == 765.0):
            continue
        print('{} : {} : {}'.format(imgpath, classes[color.sum()], count))
        cv2.putText(img, "{}: {}".format(classes[color.sum()], count), (10, 25+line), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        line += 15

    cvimshow(imgpath, img)
    print ('-' * 30)

# Ref :
# https://stackoverflow.com/questions/45043617/count-the-number-of-objects-of-different-colors-in-an-image-in-python