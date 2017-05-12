# *************************************************
# ******** Facial Detection and Recognition *******
# ************ Merav Joseph 200652063 *************
# ************* Shir Amir 209712801 ***************
# *************************************************

import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

# program constants
SIZE_X = 128
SIZE_Y = 128
FACE_PATH = r"utility\haarcascade_frontalface_default.xml"
EYE_PATH = r"utility\haarcascade_eye.xml"
FACE_CLASSIFIER = cv2.CascadeClassifier(FACE_PATH)
EYE_CLASSIFIER = cv2.CascadeClassifier(EYE_PATH)
IMG_LENGTH = SIZE_X * SIZE_Y

def find_faces(img, debug=False, scale_factor=1.3, min_neighbors=5):
    """ Returns resized faces ROI. If no face detected, returns empty list.
    :param img: the image to face - detected. 
    :param debug: show extra data in debugging mode.
    :param scale_factor: Viola-Jones parameter.
    :param min_neighbors: Viola-Jones parameter.
    :return faces: list of detectd faces' ROI resized to SIZE_X x SIZE_Y
    :return faces_coor: coordinates of each detected face in original image
    """

    assert img.ndim == 2, "Image must be 2 dimensional."
    assert img.dtype == np.uint8, "Image must be of type uint8."

    # detect faces in image
    detected_faces = FACE_CLASSIFIER.detectMultiScale(img, scale_factor, min_neighbors)
    faces = []
    faces_coor = []

    # align by resizing all faces to same size
    for (x, y, w, h) in detected_faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        roi = img[y:y + h, x:x + w]
        eyes = EYE_CLASSIFIER.detectMultiScale(roi)
        resized_roi = cv2.resize(roi, (SIZE_X, SIZE_Y), interpolation=cv2.INTER_LINEAR)
        faces.append(resized_roi)
        faces_coor.append((x,y,w,h))

    if debug:
        img_debug = img.copy()
        img_debug = cv2.cvtColor(img_debug, cv2.COLOR_GRAY2RGB)
        for (x, y, w, h) in detected_faces:
            cv2.rectangle(img_debug, (x, y), (x + w, y + h), (255, 0, 0), 2)
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(img_debug, (ex+x, ey+y), (ex + x + ew, ey + y + eh), (0, 255, 0), 2)
        cv2.imshow("face_detection", img_debug)
        cv2.waitKey()

    return faces, faces_coor

def compute_eigenfaces(faces_mat, eigenfaces_num=100):
    """ Computes the eigenfaces.
    :param faces_mat: contains all the faces in format height x width x num_of_faces
    :param eigenfaces_num: number of eigenfaces to output
    :return eigenfaces: the eigenfaces of faces_mat
    :return faces_proj: the projection of each face on the eigenfaces
    """

    # Obtain normalized face vectors
    faces_vecs = faces_mat.reshape((-1, faces_mat.shape[-1])).astype(np.float32)
    faces_vecs_norm = faces_vecs - np.mean(faces_vecs, axis=1).reshape((-1, 1))

    # Compute SVD decomposition on the noramlized face vectors
    [u, s, v] = np.linalg.svd(faces_vecs_norm)
    eigenvecs = u[:, :eigenfaces_num]
    eigenfaces = eigenvecs.reshape((faces_mat.shape[0], faces_mat.shape[1], -1))

    # Project input on eigenfaces
    faces_proj = np.dot(faces_vecs_norm.T, eigenvecs)
    return eigenfaces, faces_proj

def mean_eigenvecs(faces_mat, eigenfaces, labels):
    """ Compute the mean projection of each person on the eigenvectors.
    :param faces_mat: contains all the faces in format height x width x num_of_faces
    :param eigenfaces: the eigenfaces
    :param labels: the labels of all people in training set
    :return mean_vecs: the mean projection of each person on the eigenvectors 
    :return labels_unique: all the names of people in the training set 
    :return cov_mat: the covariance matrices of each mean projection 
    """
    faces_vecs = faces_mat.reshape((-1, faces_mat.shape[-1])).astype(np.float32)
    eigenvecs = eigenfaces.reshape((-1, eigenfaces.shape[-1]))

    # Compute distance of each person to the eigenfaces
    labels_unique = np.unique(labels)
    mean_list = []
    cov_list = []
    for l in labels_unique:
        cur_inds = np.flatnonzero(np.array(labels) == l)
        cur_faces_vecs = faces_vecs[:, cur_inds].astype(np.float32)
        cur_face_mean = np.mean(cur_faces_vecs, axis=1)
        cur_face_mean -= np.mean(cur_face_mean)
        proj_mean = np.dot(cur_face_mean, eigenvecs)
        mean_list.append(proj_mean)
        proj = np.dot(cur_faces_vecs.T, eigenvecs)
        cov_list.append(np.cov(proj.T))
    mean_vecs = np.vstack(mean_list).T
    cov_mat = np.dstack(cov_list)
    return mean_vecs, labels_unique, cov_mat

def compute_mahal_dist(img, eigenvecs, mean_vecs, cov_mat):
    """ compute the mahalanobis distance between a new face and people in training set.
    :param img: the new face
    :param eigenvecs: the eigenvectors
    :param mean_vecs: the mean projection of each training set
    :param cov_mat: the covariance matrix of each training set
    :return dist_mahal: the list of mahalanobis distances between img and each person
    """

    assert img.ndim == 2, "Image must be 2 dimensional."
    assert img.dtype == np.uint8, "Image must be uint8."
    assert img.size == eigenvecs.shape[0], "Number of pixels must be as eigenvecs first dimension."
    assert eigenvecs.shape[1] == mean_vecs.shape[0], "Number of eigenvectors should match."
    assert mean_vecs.shape[1] == cov_mat.shape[2], "Number of people should match in mean_vecs and cov_mat."

    # Reshape image to vector and project to eigenvecs space
    vec = img.flatten().astype(np.float32)
    vec -= np.mean(vec)
    vec_p = np.dot(vec, eigenvecs)
    vec_p = vec_p[:, np.newaxis]

    # Compute Mahalanobis Distance
    dist_mahal = []

    for person_ind in range(mean_vecs.shape[1]):
        # Assuming only diagonal covariance
        cov_inv = np.diag(1/np.diag(cov_mat[:, :, person_ind]))
        mu = mean_vecs[:, person_ind]
        tmp = vec_p - mu[:, np.newaxis]
        d = np.sqrt(tmp.T.dot(cov_inv).dot(tmp))
        dist_mahal.append(d[0][0])

    return dist_mahal

def classify(mahal_dist):
    """ Match an image to its most similar face in the database
    :param mahal_dist: the Mahalanobis distances if the image to each person in the database.
    :return: index of best fitting distance and ratio_test
    """

    # Sort by Mahalnobis distance
    sorted_inds = np.argsort(np.array(mahal_dist))

    # choose the best matched label
    min_dist_mahal = mahal_dist[sorted_inds[0]]

    # Low ratio_test is wanted ( = good confidence in identification)
    ratio_test = mahal_dist[sorted_inds[0]] / mahal_dist[sorted_inds[1]]
    return sorted_inds[0], ratio_test

def load_dataset(db_dir, verbose=False):
    """ loads all the images and labels in the data base
    :param db_dir: the database dir
    :param verbose: show log or not
    :return images: list of loaded images (only face RIO)
    :return labels: list of correlating labels
    """
    assert os.path.exists(db_dir), "DB folder %s doesn't exist" % db_dir

    people_folders = os.listdir(db_dir)
    labels = []
    images = []

    for person in people_folders:
        img_files = os.listdir(os.path.join(db_dir, person))
        for img_fn in img_files:
            if verbose:
                print("Reading: %s | %s | " % (person, img_fn), end="")
            img = cv2.imread(os.path.join(db_dir, person, img_fn), cv2.IMREAD_GRAYSCALE)
            faces, _ = find_faces(img)
            if len(faces) == 1:
                if verbose:
                    print("face found")
                images.append(faces[0])
                labels.append(person)
            else:
                if verbose:
                    print("no face found")
    return images, labels
