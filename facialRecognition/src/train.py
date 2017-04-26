# *************************************************
# ******** Facial Detection and Recognition *******
# ************ Merav Joseph 200652063 *************
# ************* Shir Amir 209712801 ***************
# *************************************************

import numpy as np
import cv2
import os
import pandas as pd
import eigenfaces as ef

def add_training_set(dir_path):
    """ add another face to the database
    :param dir_path: path to the directory that contains set of images.
    directory's name must be the label's name.
    """
    name = dir_path.split('/')[-1]
    file_list = os.listdir(dir_path)
    faces_db = []

    # for each image of the subject
    for f in file_list:
        img = cv2.imread('%s/%s' % (dir_path, f))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        detected_face = ef.FACE_CASCADE.detectMultiScale(gray, 1.3, 5)
        # detect the face in the image as aligning
        for (x, y, w, h) in detected_face:
            roi_gray = gray[y:y+h, x:x+w]
            resized_img = cv2.resize(roi_gray, (ef.SIZE_X, ef.SIZE_Y), interpolation=cv2.INTER_LINEAR)
            # add the aligned image and a column in faces_db
            faces_db.append(resized_img)
        cv2.imshow('face decetion', img)

    #detect eigenfaces
    ef.create_eigenfaces(faces_db)

    raw_data = {'eigenvals': [18, 17], 'eigenvecs': [[2, 3, 5, 43], [5, 7, 39,]]}
    df = pd.DataFrame(raw_data, columns=['eigenvals', 'eigenvecs'])
    df.to_csv('../image_data/%s.csv' % name)

if __name__ == "__main__":

    # assuming db_path contains subdirectories for each set of images per person
    db_path = '../images'
    dir_list = os.listdir(db_path)

    for d in dir_list:
        add_training_set('%s/%s' % (db_path, d))
