# *************************************************
# ******** Facial Detection and Recognition *******
# ************ Merav Joseph 200652063 *************
# ************* Shir Amir 209712801 ***************
# *************************************************

import numpy as np
import cv2
import eigenfaces as ef
import os

img_name = '../images/s1/1.png'
img = cv2.imread(img_name)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = ef.FACE_CASCADE.detectMultiScale(gray, 1.3, 5)
eigen_files = os.listdir('../eigenfaces')
eigen_files = eigen_files[0:1]

faces_mat = np.zeros((1, ef.IMG_LENGTH))

for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
    roi_gray = gray[y:y+h, x:x+w]
    resized_img = cv2.resize(roi_gray, (ef.SIZE_X, ef.SIZE_Y), interpolation=cv2.INTER_LINEAR)
    faces_mat = np.vstack((faces_mat, resized_img.reshape(1, ef.IMG_LENGTH)))
    roi_color = img[y:y+h, x:x+w]
    eyes = ef.EYE_CASCADE.detectMultiScale(roi_gray)

    for (ex, ey, ew, eh) in eyes:
        cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)

# delete dummy row from matrix
faces_mat = np.delete(faces_mat, 0, 0)

for f in eigen_files:
    eigenvecs = np.genfromtxt('../eigenfaces/%s' % f, delimiter=',')
    mean = eigenvecs[0]
    mean = mean.reshape((1,ef.IMG_LENGTH))
    eigenvecs = np.delete(eigenvecs, 0, 0)
    img_reconstruction = cv2.PCAProject(faces_mat, mean, eigenvecs)
