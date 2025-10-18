# Predictive-Crime-CNN-
This Convolutional Neural Network predicts the locations of clusters of violent crime in Los Angeles for the upcoming week. It was built using Python/TensorFlow and is built on top opf a dataset comprised of data gathered from official crime data published by the city of LA and weather data gathered from National Weather Service's online repository of weather station data.

## Project Overview
This model is an example of a multimodal deep learning model that combines spatial-temporal data (in this case, crime organized via coordinates/grid squares and weather over time) from multiple sources. Overall, the dataset is comprised of past crime data, weather data, and holiday data. 
The model treats LA as a 60/75 grids for a total of 4500 grid squares. For each week, data is aggregated and totaled per grid square (crime counts per grid square, weather events per grid square, etc.) and the goal is to use WINDOW_SIZE weeks of data to predict the crime counts per grid square for up to one week in the future.
This is a supervised learning and regression task: the data is labeled and the model is predicting continuous values -- crime counts. TensorFlow and Keras are used to build a Convolutional Neural Network (CNN) with two branches (one for crime and one for weather), which merge to make predictions. A CNN is ideal for this task because the data has spatial structures similar to that of an image, as well as a temporal sequence (weeks in a row). 
This code is divided into three main sections, and I will break each down in detail and explain how they work. I will also be providing another README and code samples that were used to create the dataset, and those deserve their own, individual documents and explanations because the dataset required a lot of time and effort to gather, prepare and put together. 

## Code Explanation

### Section 1: Building the Neural Network
This section defines the architecture of the model using Keras, which allows for multiple inputs and outputs. 
