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
EYES_PATH = "utility/haarcascade_eye.xml"

FACE_CASCADE = cv2.CascadeClassifier(FACE_PATH)
EYE_CASCADE = cv2.CascadeClassifier(EYES_PATH)

IMG_LENGTH = SIZE_X * SIZE_Y

def create_eigenfaces(faces_mat, amount=10):
    """ calculates eigenvectors
    :param faces_mat: a matrix in wich each row is an image vector
    """
    img_num = len(faces_mat)
    # t contains each img as a column in it
    t_mat = faces_mat.copy()
    average_face = np.mean(faces_mat, axis=0)
    average_mat = np.matlib.repmat(average_face, img_num, 1)
    t_mat = t_mat - average_mat
    mean, eigenvecs = cv2.PCACompute(t_mat, np.array([]))

    for i in range(np.shape(eigenvecs)[0] - 1):
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        cax = ax.imshow(eigenvecs[i,:].reshape(SIZE_X,SIZE_Y), cmap='gray')
        ax.set_title("EigenFace")
        plt.show()
    return np.vstack((mean, eigenvecs))

def get_mahalanobis_params(faces_mat, mean, eigenvecs):
    """ compute the mean projection of each person on the eigenvectors
    :param faces_mat: a matrix in which each row is an image vector
    :param mean: the mean image vector
    :param eigenvecs: a matrix in which each row is an eigenvector
    """
    # project all images in faces_mat on eigenvectors
    img_proj = cv2.PCAProject(faces_mat, mean, eigenvecs)
    mean_proj = np.mean(img_proj, axis=0)
    cov_mat = np.cov(img_proj)
    inv_cov_mat = np.linalg.inv(cov_mat)
    return np.vstack((mean_proj, inv_cov_mat))
