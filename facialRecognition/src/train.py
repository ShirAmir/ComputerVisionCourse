# *************************************************
# ******** Facial Detection and Recognition *******
# ************ Merav Joseph 200652063 *************
# ************* Shir Amir 209712801 ***************
# *************************************************

import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import eigenfaces as ef

def add_training_set(dir_path):
    """ adds a data set to the training set
    :param dir_path: new data set's directory path
    """

    # Load the datasets
    images, labels = ef.load_dataset("../images/train_data_set", True)
    new_images, new_labels = ef.load_dataset(dir_path, True)
    images.extend(new_images)
    labels.extend(new_labels)

    faces_mat = np.dstack(images)

    # Compute eigenfaces
    eigenfaces, faces_proj = ef.compute_eigenfaces(faces_mat)

    # Reshape to vectors
    mean_vecs, labels_unique, cov_mat = ef.mean_eigenvecs(faces_mat, eigenfaces, labels)

    # Save eigenfaces
    print("Saving train data")
    np.savez("train_data", eigenfaces=eigenfaces, faces_proj=faces_proj,
             mean_vecs=mean_vecs, labels_unique=labels_unique, cov_mat=cov_mat)


def run_training():

    # Load the datasets
    images, labels = ef.load_dataset("../images/train_data_set", True)

    faces_mat = np.dstack(images)

    # Compute eigenfaces
    eigenfaces, faces_proj = ef.compute_eigenfaces(faces_mat)

    # Reshape to vectors
    mean_vecs, labels_unique, cov_mat = ef.mean_eigenvecs(faces_mat, eigenfaces, labels)

    # Save eigenfaces
    print("Saving train data")
    np.savez("train_data", eigenfaces=eigenfaces, faces_proj=faces_proj,
             mean_vecs=mean_vecs, labels_unique=labels_unique, cov_mat=cov_mat)

if __name__ == "__main__":
    run_training()
