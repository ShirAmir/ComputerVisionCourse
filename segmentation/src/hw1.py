# *************************************************
# *** Semantic Segmentation from Training Image ***
# ***************** Merav Joseph *****************
# ******************* Shir Amir *******************
# *************************************************

from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
import matplotlib.pyplot as plt
import numpy as np
import cv2

def set_img_border(image, border_size, border_value = -1):
    """ confine borders so they wouldn;t be chosen as patches. 
    
    :param image: the image to be confined
    :param border_size: the thikness of the border 
    :param border_value: the value of the border pixels
    :return: 
    """
    img = image.copy()
    img[:, 0:border_size] = border_value
    img[:, -border_size:] = border_value
    img[0:border_size, :] = border_value
    img[-border_size:, :] = border_value
    return img


def extract_patches(im_rgb, im_segments, patch_size=11, sample_size=0.01):
    """ creates patches for each segment.
    :param im_rgb: the original image
    :param im_segments: the segmented image
    :param patch_size:  the edge size of the patch
    :param sample_size: fraction of sampled pixels in each fragment
    :return: 
    """
    half_patch_size = int(patch_size / 2)
    im_segments_with_border = set_img_border(im_segments, half_patch_size, border_value=-1)

    # array of unique segments numbers
    unique_segments = np.unique(im_segments)

    # loop over all segments
    result = []
    for seg_num in unique_segments:
        # Find all pixels coordinates in the segment
        (i, j) = np.where(im_segments_with_border == seg_num)

        patches = []
        # Make sure we have at least 1 pixel in segment
        if len(i) > 0:
            # Choose randomly pixels from segment
            sample_indices = np.random.random_integers(0, len(i)-1, np.int64(np.ceil(sample_size*len(i))))

            # Construct patches
            for patch_ind, pix_ind in enumerate(sample_indices):
                print('i=%d, j=%d' % (i[pix_ind], j[pix_ind]))
                roi = im_rgb[(i[pix_ind] - half_patch_size):(i[pix_ind] + half_patch_size + 1),
                            (j[pix_ind] - half_patch_size):(j[pix_ind] + half_patch_size + 1), :]
                patches.append(roi)
        result.append(patches)
    return result

if __name__ == "__main__":
    # PARAMS
    frag_amount = 10
    patch_size = 9
    train_img_path = "../images/martin_train.png"
    labels_img_path = "../images/martin_train_labels.png"
    test_img_path = "../images/martin_test.png"

    # Load images
    train_img = cv2.imread(train_img_path)
    labels_img = cv2.imread(labels_img_path, cv2.IMREAD_GRAYSCALE)
    test_img = cv2.imread(test_img_path)

    # Extract train patches from each label
    train_labels_patches = extract_patches(train_img, labels_img, patch_size)
    labels_nums = range(len(train_labels_patches))

    # Compute SLIC fragmentation
    # Each pixels contains the key of its' fragment
    fragments = slic(img_as_float(test_img), n_segments = frag_amount, sigma = 5)

    # Show SLIC output
    fig = plt.figure("superpixel fragmentation")
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(mark_boundaries(img_as_float(cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)), fragments))
    plt.axis("off")

    # Extract test patches from fragments
    fragments_patches = extract_patches(test_img, fragments, patch_size)
    fragments_nums = range(len(fragments_patches))

    # Compute distance
    distance = np.zeros((len(fragments_nums), len(labels_nums)), dtype = np.float32)
    normalize = float(patch_size * patch_size * 3 * 255)  # normalize factor
    for frag_key in fragments_nums:
        frag_patches = fragments_patches[frag_key]
        for label_key in labels_nums:
            label_patches = train_labels_patches[label_key]
            label_patches_mat = np.stack(label_patches)
            cost_patches = []
            for frag_p in frag_patches:
                diff_sqr = ((label_patches_mat - frag_p)/normalize)**2
                ssd = np.sum(diff_sqr.reshape((label_patches_mat.shape[0], -1)), axis=1)
                min_ssd = np.min(ssd)
                cost_patches.append(min_ssd)
            distance[frag_key, label_key] = np.median(cost_patches)

    # Cost matrix plot
    fig = plt.figure("Cost Matrix")
    ax = fig.add_subplot(1, 3, 1)
    cax = ax.imshow(labels_img)
    ax.set_title("Train Labels")
    ax = fig.add_subplot(1, 3, 2)
    cax = ax.imshow(fragments)
    ax.set_title("Test Fragments")
    ax = fig.add_subplot(1, 3, 3)
    cax = ax.imshow(distance)
    ax.set_title("Cost matrix")
    ax.set_xlabel("Train Labels")
    ax.set_ylabel("Test Fragments")
    fig.colorbar(cax, label="Cost")
    fig.tight_layout()

    plt.show()