# *************************************************
# ******** Facial Detection and Recognition *******
# ************ Merav Joseph 200652063 *************
# ************* Shir Amir 209712801 ***************
# *************************************************
import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
CURRENT_PATH = os.path.dirname(os.path.realpath(__file__))
sys.path.append(CURRENT_PATH)
from eigenfaces import find_faces

def get_statistics(positive_dir, pos_num, negative_dir, neg_num):
    """
    
    :param positive_dir: path for the positive examples directory
    :param pos_num: number of faces in the positive directory
    :param negative_dir: path for the negative examples directory
    :param neg_num: number of faces in the negative directory
    :return: 
    """
    pos_faces = []
    neg_faces = []
    dir_list = os.listdir(positive_dir)
    for fn in dir_list:
        cur_fn = os.path.abspath(os.path.join(positive_dir, fn))
        im = cv2.imread(cur_fn, cv2.IMREAD_GRAYSCALE)
        faces = find_faces(im, debug=False)
        if len(faces):
            pos_faces.append(faces)

    dir_list = os.listdir(negative_dir)
    for fn in dir_list:
        cur_fn = os.path.abspath(os.path.join(negative_dir, fn))
        im = cv2.imread(cur_fn, cv2.IMREAD_GRAYSCALE)
        faces = find_faces(im, debug=False)
        if len(faces):
            neg_faces.append(faces)

    true_positive = len(pos_faces)
    false_positive = len(neg_faces)
    true_negative = neg_num - len(neg_faces)
    false_negative = pos_num - len(pos_faces)

    precision = float(true_positive) / (true_positive + false_positive)
    recall = float(true_positive) / (true_positive + false_negative)

    return precision, recall



if __name__ == "__main__":
    parent_path = os.path.join(CURRENT_PATH, "..", "face_detection_test_images")
    # Calculate precision and recall for the easy set
    precision_easy, recall_easy = get_statistics(os.path.join(parent_path, "faces_easy"), 55,
                                      os.path.join(parent_path, "non_faces_easy"), 52)

    # Calculate precision and recall for the hard set
    precision_hard, recall_hard = get_statistics(os.path.join(parent_path, "faces_hard"), 79,
                                      os.path.join(parent_path, "non_faces_hard"), 55)

    print("easy set results: precision =  %.3f, recall = %.3f" % (precision_easy, recall_easy))
    print("hard set results: precision =  %.3f, recall = %.3f" % (precision_hard, recall_hard))

