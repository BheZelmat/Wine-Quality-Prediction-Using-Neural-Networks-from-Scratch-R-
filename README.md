# Wine Quality Prediction Using Neural Networks

![alt text](https://github.com/BheZelmat/Wine-Quality-Prediction-Using-Neural-Networks-from-Scratch-R-/blob/main/img0.png?raw=true)

## Overview
This project aims to predict the quality of wine based on various physicochemical properties using neural network models implemented in R. The project comprises two main R script files: Main.R and NN_functions.R, and a dataset wine.csv.

## Files in the Project
###Main.R:

This is the primary script file where the neural network model is applied to the wine dataset.
It begins by sourcing the NN_functions.R file, which contains necessary functions for the neural network.
The script utilizes the tidyverse library for data manipulation and visualization.
Key processes in this file include data loading, preprocessing, model training, and evaluation.
###NN_functions.R:

This file contains a suite of functions related to neural network operations.
It includes various activation functions such as sigmoid, tanh, softmax, and an identity function.
The functions in this file are utilized in the Main.R script for building and operating the neural network model.
wine.csv:

This dataset contains the physicochemical properties of wines along with their quality ratings.
It is used as the input data for training and testing the neural network model.
##Dependencies
R Programming Language
R Packages: tidyverse, reshape2
## Setup and Execution
Ensure that R and the required packages are installed.
Place the NN_functions.R, Main.R, and wine.csv in the same directory.
Run the Main.R script in R or an R IDE. This script will automatically source the NN_functions.R file.
## Functionality
Data Processing: The Main.R script handles the loading and preprocessing of the wine.csv dataset.
Model Building and Training: Utilizes the functions defined in NN_functions.R to build and train a neural network model on the wine dataset.
## Evaluation: 
The script includes procedures for evaluating the performance of the neural network model.
## Author
B Houssem E Zelmat 
