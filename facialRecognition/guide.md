# Facial Recognition using Training Set
**Merav Joseph 200652063 & Shir Amir 209712801**

The program attemps to detect faces in an image and recognize the faces that appear in the training set.

### Algorithm & Implementation Details
Our program contains 2 main stages - Training and Testing.  
The training stage analyzes several data sets, each containing face images of a certain person. 
And on the other hand, the testing stage detects new faces in an image and classifies them into the 
labels from the training stage.

Here is what both stages do in a nutshell:  

__Train:__     
    1. Detect & align all faces in training sets.  
    2. Compute the eigenfaces on all the faces.  
    3. Compute the parameters needed for test stage.  

__Test:__  
    1. Detect & align faces in the test image.  
    2. Compute the distance between each new face and each data set.  
    3. Classify which face belongs to each data sets and which do not belong to any.  

#### Training 
During the training stage, several data sets are processed in order to achieve crucial data. 
That includes the eigenfaces, the covariance matrices and the mean projections of each person.  
The eigenfaces are acquired by applying the Principal Component Analysis 
([PCA](https://en.wikipedia.org/wiki/Principal_component_analysis)) on all the acquired faces. 
Then, we acquire the projection of all the faces in the data set on the face space.  
Afterwards, we obtain the parameters required for calculating the 
[Mahalanobis Distance](https://en.wikipedia.org/wiki/Mahalanobis_distance) between each test face and
a labeled person's set of faces. This distance is later used to classify the test faces.
The Covariance Matrix of each training set is estimated using the projected faces of each person, 
while the mean projection is also calculated using the projected faces.  
At last, all this valuable data is stored in CSV files.

#### Testing
During the testing stage, a new image is analyzed in attempt to recognize the faces in it.  
At first, we use the [Viola-Jones](https://en.wikipedia.org/wiki/Viola%E2%80%93Jones_object_detection_framework)
algorithm for fast face detection. Then, we calculate the 
[Mahalanobis Distance](https://en.wikipedia.org/wiki/Mahalanobis_distance) between each newly detected face and
each training set. The calculation is done using the data obtained in the training stage.  
We chose this measuring method because it uses the standard deviation of the cluster as a measurement unit. 
This means it considers a deviation from a tight cluster more severely than in a spreaded cluster. And is thus 
"smarter" than less sophisticated measurements.  
After that, we look for the label that minimized the distance for each test face, and assign that label to it. 
If the minimal distance overshoots a certain threshold, it is considered to be very far from all the data sets 
and is assigned the 'unknown'
label. 

### Results

### Building Instructions

### Using Instructions

#### Directory Tree

This is the tree of our project:

```
facialRecognition
├── src  
│   ├── train.py 
│   └── test.py 
└── guide.md 
```

The sub-directory `src` contains all our code.
`train.py` contains the implementation of training the set of faces, while `test.py` contains
the code that detects and recognizes faces in an image.