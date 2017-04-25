# *************************************************
# ******** Facial Detection and Recognition *******
# ************ Merav Joseph 200652063 *************
# ************* Shir Amir 209712801 ***************
# *************************************************
import numpy as np
import cv2
import matplotlib.pyplot as plt

def create_eigenfaces(faces_db):
    """ calculates eigenvectors and eigenvalus
    :param faces_db: a matrix in wich each column is an image vector
    """

    # Stacking images. each row is a pixel, each column is an image.
    faces_mat = np.dstack(faces_db)
    T = faces_mat.reshape((-1, len(faces_db)))

    # Calculate average face
    average_train_face = faces_mat.astype(float).mean(axis=2)

    # Show Results
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    cax = ax.imshow(average_train_face, cmap='gray')
    ax.set_title("Average Face")
    plt.show()