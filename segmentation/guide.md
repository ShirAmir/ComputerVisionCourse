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
│   ├── khaleesi.jpg
│   ├── martin_test.png
│   ├── martin_train.png
|   ├── martin_train_labels.png  
│   └── martin_train_segments.png
├── src  
|   ├── hw1.py  
|   └── hw1_segment_train.py
└── README.md 
```

The sub-directory `src` contains all our code. Our main program is `hw1.py`, 
while `hw1_segment_train.py` is a program we used to create the segmented samples.
Also, all the images we used for testing our program can be found in `images`. 

 

