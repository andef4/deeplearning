# Deep Learning example projects
This repository contains multiple projects I implemented as a preparation for my bachelors thesis on the Bern University of Applied Sciences.

All projects use PyTorch and are located in the notebooks/ subdirectory. Python dependencies are saved in requirements.txt. The docker/ directory contains a Dockerfile to build a docker container containing PyTorch, CUDA and jupyter notebook.

## MNIST
An implementation of the MNIST handwritten digit classification dataset.

## Music
Detect the genre of a song. Use music_prepare.ipybn to generate a dataset from a local music library, then music.ipynb to train the classifier.

## Icons
Detects 2D shapes (tetris stones) in real time from a webcam. This is a multi-label classifier because it can detect multiple shapes under the webcam.

## Shapes3D
This is a unfinished project. It currently downloads and extracts 3D model files (*.stl) from thingiverse. The next step would be to generate renderings from different angles of the 3D models and then use these as input for the network.

One idea for a classifier would be to give the network three images of the same model from different angles and as a second input another image from the same or from another model. Then the network has to decide if the second input shows the same image as the three sample images.

## Tetris
A very early attempt to create a Tetris bot. Based on code by Kevin Chabowski  https://gist.github.com/silvasur/565419/d9de6a84e7da000797ac681976442073045c74a4, adjusted to work inside a jupyter notebook. Not bot code yet.
