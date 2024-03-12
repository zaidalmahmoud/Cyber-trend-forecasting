The file **BMTGNN.py** contains a script for evaluating the performance of the B-MTGNN model on unseen data. The model uses 10 iterations by default to approximate a Bayesian model. For using other iterations such as 20,30,40 or 50, please adjust the global variable **runs** in line 21. The experiment is repeated five times and the average errors (RSE and RAE) are computed and printed on the terminal. The errors can be copied and saved to the file **outb10.txt** (If 20 iterations used, the file name is  **outb20.txt**).

In each of the 5 experiments, the script performs random search with 30 iterations to optimise the set of hyperparameters. That is, it finds the model with the least validation error. The best model is finally saved to the file **modelb10.pt** and used to forecast the unseen data.

The directory contains the results of five different Bayesian models, each uses different number of iterations in the range 10-50.

