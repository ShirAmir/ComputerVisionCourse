# ComputerVisionCourse
This repository contains assignments from the Computer Vision Course by Prof. Hagit Hel-Or. 
The course's webpage can be found [here](http://cs.haifa.ac.il/hagit/courses/CV).

### Workspace
We use **Anaconda 2.4.0** Python Distribution on **Windows 10**.
Just download `Anaconda3-2.4.0-Windows-x86_64.zip` from [this link](https://repo.continuum.io/archive/.winzip/ "zipped Windows installers"), unzip and install.

Also, we use **Python 3.5.3** with the following primary modules:
  * **OpenCV 3.1.0** for computer vision algorithms.
  * **Numpy 1.12.1** for vectorail manipulation. 
  * **Matplotlib 2.0.0** for data visualizing.
  * **Tk 8.5.18** for creating a GUI.

#### What if I already have Anaconda?
Easy! Simply create a new environment within your Anaconda distribution.
Conda enables us to create, delete and configure different environments, with different python versions and packages.
Here we will show how to create an environment to match this repository's prerequisits. 
 1. Type `python --version` in your conda prompt. If it output's 3.5.3 then you can skip this whole paragraph. :smile:
 2. In your conda prompt type `conda create -n cvCourse python=3.5.3 numpy`. This command creates an environment named *cvCourse* with the python 3.5.3 and numpy.
 3. Activate your new environment by typing `activate cvCourse` in your conda prompt. At this point the name of your new environment should be displayed within parethesis on the left side of your path in the conda prompt. You can check the environment's python version by typing `python --version` in your conda prompt. It should now output python 3.5.3.
 4. Now you must add relevant packeges to your environment as specified in the next paragraph.
 5. Finaly once done with the environment deactivate it by typing `deactivate cvCourse` in your conda prompt. At this point the name of your new environment within the prenthesis should disappear from the left side of your path in the conda prompt.
 
***NOTE: make sure you activate your environment before trying to run the code!***

After setting up Anaconda (or an anaconda environment), make sure all the following modules are up-to-date:
  * Cython >= 0.23
  * Six >= 1.7.3
  * SciPy >= 0.17.0
  * Numpydoc >= 0.6
  * NetworkX >= 1.8
  * Pillow >= 2.1.0
  * PyWavelets >= 0.4.0
  * Scikit-image >= 0.12.3

This can be done by type `conda list` in the conda prompt and observe the different modules.  
In case one needs to be installed or updated, simply type `conda module_name install` in the conda prompt. 
OpenCV is installed by typing `conda install -c menpo opencv3=3.1.0` in the conda prompt.

### Assignments
This course requires 5 assignments:
  1. [Segmentation](/segmentation/guide.md)
  2. [Facial Recognition](/facialRecognition/guide.md)
  3. [Motion Tracking](/bugTracker/guide.md)
