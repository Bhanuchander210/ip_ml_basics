## Object Detection
---

> Finding Different objects in an Image and recognize them

#### Goals :
---

- Finding the list of bounding boxes \[(x, y)-coordinates\] for each known objects.
- Find all class labels associated with each bounding boxes.
- The probability/confidence score associated with each bounding box and class label.

#### Common Methods to find the Objects :
---

- Sliding Window.
- Image Pyramid.

#### Object Detection - Methods
---

- R-CNN
- Fast R-CNN
- SSD (Single Shot Detectors)
- Faster R-CNN
- Mask R-CNN
- YOLO (You Only Learn Once)

###### R - CNN
---

> Region Based - Convolutional Neural Network.

- **Selective Search mechanism** used to find more object regions (Region of Interest) around ~2000.
- These regions are independently processed for image classification.
- RCNN uses simple linear regression to find the tighter **bounding boxes** of the object. Here sub regions of a single object are
the inputs.
- Normal **SVM** used for classification.


###### Fast R-CNN
---

> Speeding Up and Simplified form of RCNN

Why RCNN failed / need optimization ?

- Normal R- CNN needs to train three various modules for the operation
    - A CNN to generate image features.
    - The classifier which predicts the image class
    - The Regression module which places the correct border over object.
- That's why it is slow. and doing classification on same image again and again.


**Solution :**

- Using **Region of Interest Pooling**
- Placing these various moudles (above discussed) in a single network.    
- Normal SVM is replaced with **Soft Max Layer**.

###### Faster R-CNN
---

> Speeding Up Region Proposal

- Not Using **Selective Search**.
- Same CNN Used to do Region proposals.
- only one CNN needs to be trained for total process instead of **Three training**.
- Sliding window used for feature extracting in ROI-Proposal CNN.
- **Anchor Boxes** (k-boxes) known as common boxes for an object are passed to the next process Fast-RCNN bounding
box technique.

###### MASK - R-CNN
---

- Above all techniques are having difficulties to separate the same class overlapped objects.
- So we are in need of **Pixel level segmentation**. 
- By adding another one branch to the Faster-RCNN with a **binaryMask Matrix**.
- RoiAlign - Realigning RoIPool to be More Accurate means (Due to the process some misalignment will occur while detecting objects.)
- After generating these **Mask layer** we can accurately find the borders of the image.

**Segmentation :**

- Semantic Segmentation
    - It can't find the boundaries of the same class objects , means overlapped objects.
- Instance Segmentation
    - It predicts the boundaries as well using mask matrix.


#### References:
---

- [Gentle Intro - Object Detection](https://www.pyimagesearch.com/2018/05/14/a-gentle-guide-to-deep-learning-object-detection/)
- [Intro and History of Regional-CNN - Mask-RCNN](https://blog.athelas.com/a-brief-history-of-cnns-in-image-segmentation-from-r-cnn-to-mask-r-cnn-34ea83205de4)
- [R-CNN, Faster R-CNN and YOLO](https://towardsdatascience.com/r-cnn-fast-r-cnn-faster-r-cnn-yolo-object-detection-algorithms-36d53571365e)