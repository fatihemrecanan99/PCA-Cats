# PCA-Cats
CS464 Introduction to Machine Learning Homework 2
Cat Image Analysis with PCA
Without utilizing any machine learning tools, this homework task uses Principal Component Analysis (PCA) to evaluate cat picture data. The Animal Faces dataset, which contains 5653 cat images, was utilized in this study.

Dependencies
	Python 3.8+
	NumPy
	Pillow (PIL)

How to Run

Just Run the main script.

Description of the Code	
The dataset includes cat zip file. When you run main script, it automatically reads the cat zip file.
	The script will take the following actions:
	The cat photos should be loaded and prepared by being reduced in size to 64x64 pixels and flattened into 4096x3 matrices.
	To get the first 10 main components, do PCA on the R, G, and B color channels.
	Determine the bare minimum number of principle components needed to obtain at least 70% PVE for all channels by computing the Proportion of Variance Explained (PVE) for 	each principal component.
	Create 10 RGB pictures that represent the eigenvectors by reshaping and normalizing the primary components.
	Use k = 1, 50, 250, 500, 1000, or 4096 main components to reconstruct a cat picture.

The outcomes will be shown in the console and saved as picture files for additional examination.

Note: Depending on the operating system, the photos may not be in the same order. In this project, it is assumed that flickr cat 000003.jpg is the second image.
