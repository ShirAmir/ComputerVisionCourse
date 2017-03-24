# *************************************************
# *** Semantic Segmentation from training image ***
# ******************* Shir Amir *******************
# ***************** Meirav Joseph *****************
# *************************************************

import numpy as np
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
from skimage import io
import matplotlib.pyplot as plt

# Load image
image = img_as_float(io.imread('khaleesi.jpg'))
                               
# find superpixeles
segments = slic(image, n_segments = 300, sigma = 5)  

# combine results with original image
fig = plt.figure("Superpixels -- %d segments" % (300))
ax = fig.add_subplot(1, 1, 1)
ax.imshow(mark_boundaries(image, segments))
plt.axis("off")

# show the plots
plt.show()   