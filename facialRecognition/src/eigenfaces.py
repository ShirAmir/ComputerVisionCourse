# *************************************************
# ******** Facial Detection and Recognition *******
# ************ Merav Joseph 200652063 *************
# ************* Shir Amir 209712801 ***************
# *************************************************
import numpy as np
import numpy.matlib
import numpy.linalg
import cv2
import matplotlib.pyplot as plt

# constants
SIZE_X = 256
SIZE_Y = 256
FACE_PATH = "utility\\haarcascade_frontalface_default.xml"
FACE_CASCADE = cv2.CascadeClassifier(FACE_PATH)
IMG_LENGTH = SIZE_X * SIZE_Y
def create_eigenfaces(faces_db, amount=10):
    """ calculates eigenvectors and eigenvalus
    :param faces_db: a matrix in wich each column is an image vector
    """
    img_num = len(faces_db)
    # t contains each img as a column in it
    t_mat = np.asarray(faces_db).flatten().reshape(img_num, IMG_LENGTH).T
    average_face = np.mean(faces_db, axis=0)
    average_mat = np.matlib.repmat(average_face.reshape(IMG_LENGTH, 1), 1, img_num)
    t_mat = t_mat - average_mat
    mean, eigenvecs = cv2.PCACompute(t_mat.T, np.array([]), maxComponents=300)

    for i in range(np.shape(eigenvecs)[0] - 1):
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        cax = ax.imshow(eigenvecs[i,:].reshape(SIZE_X,SIZE_Y), cmap='gray')
        ax.set_title("EigenFace")
        plt.show()
    return eigenvecs
