# *************************************************
# ******** Facial Detection and Recognition *******
# ************ Merav Joseph 200652063 *************
# ************* Shir Amir 209712801 ***************
# *************************************************

import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import shutil
import eigenfaces as ef

TRAIN_DIR = "../images/train_data_set"

def run_training():
    """ Training the database images """

    # Load the datasets
    images, labels = ef.load_dataset(TRAIN_DIR, True)

    faces_mat = np.dstack(images)

    # Compute eigenfaces
    eigenfaces, faces_proj = ef.compute_eigenfaces(faces_mat)

    # Reshape to vectors
    mean_vecs, labels_unique, cov_mat = ef.mean_eigenvecs(faces_mat, eigenfaces, labels)

    # Save eigenfaces
    print("Saving train data")
    np.savez("train_data", eigenfaces=eigenfaces, faces_proj=faces_proj,
             mean_vecs=mean_vecs, labels_unique=labels_unique, cov_mat=cov_mat)

def add_training_set(dir_path):
    """ adds a data set to the training set
    :param dir_path: new data set's directory path
    """

    dir_name = dir_path.split("/")[-1]
    dir_exists = not(os.path.exists("%s/%s" % (TRAIN_DIR, dir_name)))
    assert dir_exists, "The directory %s/%s already exists." % (TRAIN_DIR, dir_name)
    shutil.copytree(dir_path, "%s/%s" % (TRAIN_DIR, dir_name))
    run_training()

if __name__ == "__main__":
    run_training()