import numpy as np
import cv2

# k-Means params
segNum = 5
tryNum = 2
flags = cv2.KMEANS_RANDOM_CENTERS

# Read image
im = cv2.imread("../images/martin_train.png")

# Apply k-Means
z = im.reshape((-1, 3)).astype(np.float32)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 1, 0.1)
compactness, labels, centers = cv2.kmeans(z, segNum, None, criteria, tryNum, flags)

# Convert to uint8, and create image
center = np.uint8(centers)
res1 = center[labels.flatten()]
res2 = res1.reshape(im.shape)

# Show segmented image
cv2.imshow('k-means segmentation', res2)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Save images
# np.save("martin_train_segments.npy", labels)
cv2.imwrite("../images/martin_train_segments.png", res2)

cv2.imwrite("../images/martin_train_labels.png", labels.reshape((im.shape[0], im.shape[1])).astype(np.uint8))

# # Find unique segments
# unique_segments = np.vstack({tuple(row) for row in im_segments_train.reshape((-1, 3))})
# labels = np.zeros((im_segments_train.shape[0], im_segments_train.shape[1]), dtype=np.int64)
# for seg_ind, seg_rgb in enumerate(unique_segments):
#     pixels = np.all(im_segments_train == seg_rgb, axis=2)
#     labels[pixels] = seg_ind