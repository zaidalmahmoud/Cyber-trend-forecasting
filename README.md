# Cyber trend forecasting
This repository contains an end-to-end platform for forecasting cyber threats and pertinent technologies using graph neural network. This includes data preparation, model development, and future forecast.

## Dataset
The full constructed dataset can be found in the directory [**Dataset**](https://github.com/zaidalmahmoud/Cyber-trend-forecasting/tree/main/Dataset). More information about each feature can be found in the same directory.

## Data Preparation
The directory **Data_Preparation** contains 5 scripts for extracting features from our dataset. These features include the number of incidents for each attack type (NoI), the number of attack mentions in Elsevier abstracts (A_NoM), the number of pertinent technology mentions in Elsevier abstracts (PT_NoM), the number of tweets about armed conflict areas or wars (ACA), and the number of public holidays in each country (PH). The features are extracted and recorded on a monthly basis from July 2011 to December 2022. They can be combined to form the final dataset. For more details, please refer to the README file within the same directory.

