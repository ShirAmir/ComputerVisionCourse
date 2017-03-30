# Semantic Segmentation From a Training Image
The program receives a training image, its' segmentation and a new similar image.
The program then segments the new image according to the algorithm described in 
[this article](http://www.math.tau.ac.il/~dcor/online_papers/papers/Yaar05.pdf "Semantic Segmentation, Yaar Et al.").

### Algorithm & Implementation Details

### Results

### Building Instructions

### Using Instructions
#### Directory Tree

This is the tree of our project:

```
segmentation 
├── images
│   ├── image_test
│   ├── image_train
│   ├── image_train_labels
│   └── ...
├── src  
|   ├── hw1.py  
|   └── hw1_segment_train.py
└── guide.md 
```

The sub-directory `src` contains all our code. Our main program is `hw1.py`, 
while `hw1_segment_train.py` is a program we used to create the segmented samples.  
Also, all the images we used for testing our program can be found in `images`. 
We took most of our  testing images from  [pixabay](https://pixabay.com).
Then, we used Photoshop to create the labels. We saved the labels in the following format: 
An image with k segments 1 2 ... k is saved such that every pixel from segment x is valued by x. 


 

