import cv2
import numpy as np
from skimage import io, morphology, measure
from sklearn.cluster import KMeans

img = io.imread("du0XZ.png")

def rgb_to_hex(rgb):
    return '#%02x%02x%02x' % rgb

import matplotlib.pyplot as plt
# Prepare table
columns = ('Colors', 'Objects Count')
cell_text = []
colors = []


def cvimshow(title, img):
    cv2.imshow(title, img)
    key = cv2.waitKey(20000)
    if key == 27:
        cv2.destroyAllWindows()

rows, cols, bands = img.shape
X = img.reshape(rows*cols, bands)

kmeans = KMeans(n_clusters=5, random_state=0).fit(X)
labels = kmeans.labels_.reshape(rows, cols)

line = 1
for i in np.unique(labels):
    blobs = np.int_(morphology.binary_opening(labels == i))
    color = np.around(kmeans.cluster_centers_[i])
    count = len(np.unique(measure.label(blobs))) - 1
    if count == 0 : continue
    note = 'Color: {}  >>  Objects: {}'.format(color, count)
    cell_text.append(['  ', count])
    colors.append([rgb_to_hex(tuple(color[-3:])), 'w'])
    print note
    cv2.putText(img, note, (10, 25 + line), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (0, 0, 255), 2)
    line += 25

cvimshow('Image', img)

fig, ax = plt.subplots()
ax.axis('tight')
ax.axis('off')
the_table = ax.table(cellText=cell_text,cellColours=colors,
                     colLabels=columns,loc='center')
plt.show()