# ComputerVisionCourse
This repository contains assignments from the Computer Vision Course by Prof. Hagit Hel-Or. 
The course's webpage can be found [here](http://cs.haifa.ac.il/hagit/courses/CV).

### Workspace
We use **Anaconda 2.4.0** Python Distribution on **Windows 10**.
Just download `Anaconda3-2.4.0-Windows-x86_64.zip` from [this link](https://repo.continuum.io/archive/.winzip/ "zipped Windows installers"), unzip and install.  
Also, we use **Python 3.5.3** with the following primary modules:
  * Numpy 1.12.1
  * Opencv 3.1.0
  * Scikit-image 0.12.3
  * Matplotlib 2.0.0
  * Tk 8.5.18

After setting up Anaconda, make sure all the following modules are up-to-date:
  * Cython >= 0.23
  * Six >= 1.7.3
  * SciPy >= 0.17.0
  * Numpydoc >= 0.6
  * NetworkX >= 1.8
  * Pillow >= 2.1.0
  * PyWavelets >= 0.4.0

This can be done by type `conda list` in the conda prompt and observe the different modules.  
In case one needs to be installed or updated, simply type `conda module_name install` in the conda prompt. 
OpenCV is installed by typing `conda install -c menpo opencv3=3.1.0` in the conda prompt.
  
### Assignments
This course requires 5 assignments:
  1. [Segmentation](/segmentation/guide.md)
  2. [Facial Recognition](/facialRecognition/guide.md)
