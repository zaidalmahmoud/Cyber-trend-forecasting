The file **transformer_u.py** contains a script for training and assessing the **univariate** Transformer model's forecasting performance for 142 cyber trends, predicting 36 time-steps ahead (multistep forecast). A distinct model is constructed for each trend. Evaluation of the final results, comprising 36 times 142 values, is conducted based on RSE and RAE metrics. The forecast values are saved in **forecast_results_normalised_u.csv**. Upon completion of the script, RSE and RAE values are displayed on the terminal and can be copied and saved in **out_u.txt** for reference.

The file **transformer_m.py** contains a script for training and assessing the **multivariate** Transformer model's forecasting performance for 142 cyber trends, predicting 36 time-steps ahead (multistep forecast). A distinct model is constructed for each trend where the model uses 3 additional features besides the ground truth. These are (**NoM, ACA, and PH**). Evaluation of the final results, comprising 36 times 142 values, is conducted based on RSE and RAE metrics. The forecast values are saved in **forecast_results_normalised_m.csv**. Upon completion of the script, RSE and RAE values are displayed on the terminal and can be copied and saved in **out_m.txt** for reference.




