# Cyber trend forecasting
This repository contains an end-to-end platform for forecasting cyber threats and pertinent technologies using graph neural network. This includes data preparation, model development, and future forecast.

## Dataset
The full constructed dataset can be found in the directory [**Dataset**](https://github.com/zaidalmahmoud/Cyber-trend-forecasting/tree/main/Dataset). More information about each feature can be found in the same directory.

## Data Preparation
This directory contains 5 scripts for extracting the features of our dataset. The features include the number of incidents (NoI), the number of attacks mentions in Elsevier abstracts (A_NoM), the number of pertinent technologies mentions in Elsevier (PTs_NoM), the number of tweets about Armed Conflict Areas or Wars (ACA), and the count of public holidays in each country. All these features are extracted and recorded on a monthly basis in the period between July 2011 and December 2022. The features are finally combined together as the final dataset. For more details, please refer to the README file within the same directory.

