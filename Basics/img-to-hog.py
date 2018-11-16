# Visualize the HOG Representation of an image.

from skimage.feature import hog
from skimage import exposure
import cv2

def cvimshow(title, img):
    cv2.imshow(title, img)
    key = cv2.waitKey(5000)
    if key == 27:
        cv2.destroyAllWindows()


file = "../data/avengers/ben.jpg"

image = cv2.imread(file)
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
image= cv2.resize(image, (256, 256))

cvimshow('Actual Image', image)

fd, hogImage = hog(image, orientations=9, pixels_per_cell=(16, 16), cells_per_block=(2, 2), transform_sqrt=True,
                   block_norm="L2-Hys", visualise=True)

hog_Image = exposure.rescale_intensity(hogImage, out_range=(0, 255))
hogImage = hog_Image.astype("uint8")

cvimshow('HOG Image', hogImage)