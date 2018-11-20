# import the necessary packages
from imutils import paths
import numpy as np
import cv2
 
# load the class labels from disk
rows = open("synset_words.txt").read().strip().split("\n")
classes = [r[r.find(" ") + 1:].split(",")[0] for r in rows]
 
# load our serialized model from disk
net = cv2.dnn.readNetFromCaffe("bvlc_googlenet.prototxt",
                               "bvlc_googlenet.caffemodel")
 
# grab the paths to the input images
imagePaths = sorted(list(paths.list_images("images/")))