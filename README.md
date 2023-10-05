# Cyber trend forecasting
This repository contains an end-to-end platform for forecasting cyber threats and pertinent technologies using graph neural network. This includes data preparation, model development, and future forecast.

## Dataset
The full constructed dataset can be found in the directory [**Dataset**](https://github.com/zaidalmahmoud/Cyber-trend-forecasting/tree/main/Dataset). More information about each feature can be found in the same directory.

## Pertinent Technologies Extraction/Graph Construction
The directory **PT_Extractor** contains scripts for extracting the pertinent technologies (PTs) related to each attack type. This results in the graph construction, where each attack will be represented as a node linked through edges to its PT nodes. This is referred to as the threats and pertinent technologies graph (TPT graph). The PT extraction is achieved using two algorithms and the output from both algorithms is combined as the final output. The first algorithm (E-GPT) utilises Elsevier API along with the GPT model to obtain the PTs for each attack type. The second algorithm (D-GPT) utilises the GPT model only. For more information, please refer to the README file within the same directory.
 
## Data Preparation
The directory **Data_Preparation** contains 5 scripts for extracting the features of our dataset. These features include the number of incidents for each attack type (NoI), the number of attack mentions in Elsevier abstracts (A_NoM), the number of pertinent technology mentions in Elsevier abstracts (PT_NoM), the number of tweets about armed conflict areas or wars (ACA), and the number of public holidays in each country (PH). The features are extracted and recorded on a monthly basis from July 2011 to December 2022. They can be combined to form the final dataset. For more details, please refer to the README file within the same directory.

## Modelling and Future Forecast
The directory **B-MTGNN** contains a Python project for building the graph neural network model used for forecasting the TPT graph. Some of the scripts in this directory perform hyper-parameter optimisation, followed by training the final model using the optimal set of hyper-parameters. The final model can be used to predict the graph up to 3 years in advance. The rest of scripts utilise the built model to forecast the graph including the forecast of the attack trends, the pertinent technology trends, and the gap between them until the end of 2025. For more information, please refer to the README file in the same directory.
