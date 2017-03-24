# *************************************************
# *** Semantic Segmentation from training image ***
# ***************** Meirav Joseph *****************
# ******************* Shir Amir *******************
# *************************************************

from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
import matplotlib.pyplot as plt
import numpy as np
import cv2

# load image
image = cv2.imread("khaleesi.jpg")
height = image.shape[0]
width = image.shape[1]
fragAmount = 10;

# compute SLIC fragmentation
# segments is an image in wich all pixels of a single segment contain its' key
fragments = slic(img_as_float(image), n_segments = fragAmount, sigma = 5)

# show SLIC output
fig = plt.figure("superpixel fragmentation")
ax = fig.add_subplot(1, 1, 1)
ax.imshow(mark_boundaries(img_as_float(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)), fragments))
plt.axis("off")
plt.show()

# acquire random representatives for each fragment
sampleSize = fragAmount * 10
yCoors = np.random.randint(1, height, sampleSize)
xCoors = np.random.randint(1, width, sampleSize)
randomSample = np.vstack((xCoors, yCoors))

# loop over all fragments
for (i, fragKey) in enumerate(np.unique(fragments)):
	# construct a mask for the segment
	print ("[x] inspecting segment %d" % (i))
	mask = np.zeros(image.shape[:2], dtype = "uint8")
	mask[fragments == fragKey] = 255

	# show the masked region
	cv2.imshow("Fragment %d" % (i), cv2.bitwise_and(image, image, mask = mask))
	cv2.waitKey(0)     
