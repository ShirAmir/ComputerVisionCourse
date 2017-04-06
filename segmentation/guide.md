# Semantic Segmentation From a Training Image
The program receives a training image, its' segmentation and a new similar image.
The program then segments the new image according to the algorithm described in 
[this article](http://www.math.tau.ac.il/~dcor/online_papers/papers/Yaar05.pdf "Semantic Segmentation, Yaar Et al.").

### Algorithm & Implementation Details

### Results

### Building Instructions
Building the project is pretty simple. 
Just clone the `segmentation` directory into your computer.
Before running the program, make sure your python configuration complies with  
[README requests](../README.md "README file").  

#### GUI Doesn't Work
In some systems the GUI may not work because new versions of *PIL* library desist to contain *TK* binding files. 
In this case it's best to uninstall *PIL* and *Pillow* libraries by typing `conda remove PIL` and 
`conda remove pillow` in to the conda prompt. Later, install *Pillow* using *pip installer* by typing 
`pip install pillow` in to the conda prompt. 

### Using Instructions
1. Open the command line in `src` directory. 
2. Run our GUI by typing `python gui.py` into the command line.
    At this point the GUI window will be opened:  
3. Choose input images by pressing the correlating buttons on the left:
    * **Train Image** for choosing the trainning image.
    * **Labeled Image** for choosing the labeling of the trainning image. 
    * **Test Image** for choosing the image to be segmented. 
4. When pressing one of the aforementioned buttons the file explorer will be opened:
    Then choose the requested image. 
5. The output directory can also be altered in a similar way by pressing 'Output Directory'.
6. Configure the algorithm parameters by using the entry on the right:
    * **Amount of Fragments** configures the amount of fragments in the SLIC algorithm.
    * **Patch Size** configures the size of the patches requiered in the distance determination process.
    * **Grabcut Threshold** configures the threshold requiered for Grabcut algorithm. 
    * **Grabcut Iterations** configures the amount of iterations of Grabcut algorithm.
    * **SLIC Sigma** conrigures the sigma variable (smoothness) of the fragments in SLIC algorithm.
7. Once completed configurations press the button **GO!** to run the program.
8. Wait pateintly (running time can take about 30 seconds) 
9. The segmentation results will appear on screen and as a file in the output directory you specified earlier.

#### Directory Tree

This is the tree of our project:

```
segmentation 
├── results
├── images
│   ├── image_test
│   ├── image_train
│   ├── image_train_labels
│   └── ...
├── src  
|   ├── gui.py  
|   └── segment.py
└── guide.md 
```

The sub-directory `src` contains all our code.
`segment.py` contains the implementation of the segmentation algorithm, while `gui.py` contains
the code of our graphical interface.  
In addition, the sub-directory `results` is the default location for the outputted images. 
Also, all the images we used for testing our program can be found in `images`. 
We took most of our testing images from [pixabay](https://pixabay.com).
Then, we used Photoshop to create the labels. We saved the labels in the following format: 
An image with k segments 0 1 ... k-1 is saved such that every pixel from segment x is valued by x. 
