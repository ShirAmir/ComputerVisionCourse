# *************************************************
# ******** Facial Detection and Recognition *******
# ************ Merav Joseph 200652063 *************
# ************* Shir Amir 209712801 ***************
# *************************************************

import numpy as np
import cv2
import os
import pandas as pd

# assuming db_path contains subdirectories for each set of images per person
db_path = '..\\images'
dir_list = os.listdir(db_path)

for d in dir_list:
    file_list = os.listdir('%s\\%s' % (db_path, d))
    for f in file_list:
        img = cv2.imread('%s\\%s\\%s' % (db_path, d, f))
        cv2.imshow('face decetion', img)
    #detect eigenfaces
    raw_data = {'eigenvals': [18, 17], 'eigenvecs': [[2, 3, 5, 43], [5, 7, 39,]]}
    df = pd.DataFrame(raw_data, columns=['eigenvals', 'eigenvecs'])
    df.to_csv('../image_data/%s.csv' % d)
