# CSE 515 Multimedia and Web Databases Phase #1

## Project Description

In this project, we explore image features, vector models, and similarity/distance measures.

## Tools and Libraries

- Python
- Pandas
- PyTorch (deep learning library)
- torchvision (visual information extraction package)
- numpy, scipy (array manipulation and mathematical operations)

## Pre-requisites

To have the following pre-trained neural network model and dataset downloaded through torchvision:

- Pre-trained neural architecture: ResNet50 (with default weights)
- Dataset: Caltech 101

## Task 0

Familiarize yourself with the programming environment and the required libraries.


## Task 1

Implement a program that, given an image ID and a feature model, visualizes the image, and extracts and prints the corresponding feature descriptors. The supported feature models are:

- Color Moments
- Histograms of Oriented Gradients (HOG)
- ResNet-AvgPool-1024
- ResNet-Layer3-1024
- ResNet-FC-1000

Description:

Task 1 is responsible for extracting various feature descriptors from images. It supports multiple feature models including Color Moments, Histograms of Oriented Gradients (HOG), and different layers of the ResNet architecture.

How it Works:

    1. The program loads the Caltech 101 dataset using the torchvision library.
    2. It defines functions (color_moments_fd, hog_fd, resnet_layer_fd) to extract feature descriptors from images using different methods.
    3. Based on user input, the program allows the user to choose a feature model and an image ID.
    4. It then loads the image, applies the chosen feature model, and prints the resulting feature descriptor.

How to Use Task 1:

python3 task1.py

## Task 2

Implement a program that extracts and stores feature descriptors for all images in the dataset.

Description:

Task 2 is responsible for extracting and storing feature descriptors for all images in the dataset.

How it Works:

    1. Similar to Task 1, the program loads the Caltech 101 dataset using torchvision.
    2. It uses functions (color_moments_fd, hog_fd, resnet_layer_fd) for feature extraction implemented in task1.
    3. It then iterates through all images in the dataset, applies all the feature models, and stores the resulting feature descriptors in a CSV file.

How to Use Task 2:

python3 task2.py

## Task 3

Implement a program that, given an image ID and a value "k", returns and visualizes the most similar k images based on each feature model.

Description:

Task 3 is responsible for calculating similarity scores between images based on their feature descriptors and return top K images

How it Works:

    1. The program loads the Caltech 101 dataset and the previously generated feature descriptors from the CSV file.
    2. It defines a function (cosine_similarity) to calculate cosine similarity between two vectors.
    3. For each query image, it calculates the similarity scores with all other images in the dataset using different feature models.
    4. It then selects the top-k similar images based on the calculated scores.

How to Use Task 3:

python3 task3.py