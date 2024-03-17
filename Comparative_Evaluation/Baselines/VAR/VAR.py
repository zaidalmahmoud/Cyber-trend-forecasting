import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import torch
import sys
import math
import random
import numpy as np
from statsmodels.tsa.api import VAR
from collections import defaultdict
import csv

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

fixed_seed = 123
set_random_seed(fixed_seed)



def calculate_rrse(test, predict):
    sum_squared_diff = torch.sum(torch.pow(test - predict, 2))
    root_sum_squared = torch.sqrt(sum_squared_diff)
    
    test_s = test
    mean_all = torch.mean(test_s, dim=0)
    if test_s.dim()>1:
        diff_r = test_s - mean_all.expand(test_s.size(0), test_s.size(1))
    else:
        diff_r = test_s - mean_all.expand(test_s.size(0))

    sum_squared_r = torch.sum(torch.pow(diff_r, 2))
    root_sum_squared_r = torch.sqrt(sum_squared_r)
    
    rrse = root_sum_squared / root_sum_squared_r
    return rrse.item()

def calculate_rae(test, predict):
    sum_absolute_diff = torch.sum(torch.abs(test - predict))
    
    test_s = test
    mean_all = torch.mean(test_s, dim=0)
    if test_s.dim()>1:
        diff_r = test_s - mean_all.expand(test_s.size(0), test_s.size(1))
    else:
        diff_r = test_s - mean_all.expand(test_s.size(0))

    sum_absolute_r = torch.sum(torch.abs(diff_r))
    
    rae = sum_absolute_diff / sum_absolute_r
    return rae.item()


#by Zaid et. al
#builds the graph of threats and pertinent technologies
def build_graph():
    # Initialize an empty dictionary with default value as an empty list
    graph = defaultdict(list)

    # Read the graph CSV file
    with open('data/graph.csv', 'r') as f:
        reader = csv.reader(f)
        # Iterate over each row in the CSV file
        for row in reader:
            # Extract the key node from the first column
            key_node = row[0]
            # Extract the adjacent nodes from the remaining columns
            adjacent_nodes =  [node for node in row[1:] if node]#does not include empty columns
            
            # Add the adjacent nodes to the graph dictionary
            graph[key_node].extend(adjacent_nodes)
    print('Graph loaded with',len(graph),'attacks...')

    # Initialize an empty list for column names
    columns = []

    # Read the CSV file of the dataset
    with open('data/data.csv', 'r') as f:
        reader = csv.reader(f)
        # Read the first row
        col = [c for c in next(reader)]
        col = col[1:] # remove date column

    # Print the column names list
    print(len(col), 'columns loaded...')
    

    #create adjacency matrix
    adj = torch.zeros((len(col), len(col)))

    for i in range(adj.shape[0]):
        if col[i] in graph:
            for j in range (adj.shape[1]):
                if col[j] in graph[col[i]]:
                    adj[i][j]=1
                    adj[j][i]=1
                    print(col[i],col[j])
    
    print('Adjacency created...')

    return adj

    
def getFeatures(col,adj):

    f=[]
    if col < 16:  # If column is less than 16 (attack column), append mention to the attack (NoM)
        f.append(col + 16) #mention of the attack (NoM)
        f.append(42) #wars (ACA)
        f.append(43) #public holidays (PH)
        for j in range(len(adj)):
            if adj[col][j]==1:
                f.append(j) #pertinent technology (PAT)

    elif col <42: #if column is a mention, append wars, public holidays, and any pertinent technology
        f.append(42) #wars (ACA)
        f.append(43) #public holidays (PH)
        for j in range(len(adj)):
            if adj[col][j]==1:
                f.append(j) #pertinent technology (PAT)

    elif col<44: #wars or holiday
        f.append(0) #append any possibly relevant column for VAR to work (VAR should be multivariate)

    elif col>43: #if column is a pertinent technology, append relevant attacks
        for j in range(len(adj)):
            if adj[col][j]==1:
                f.append(j) #attack
                            
    return f



def main(e):
    # Read the dataset from CSV file
    df = pd.read_csv('data/sm_data.csv', header=None)

    # Take the first 102 rows as training data and the rest as test data
    train_data = df.iloc[:66, :]
    valid_data = df.iloc[66:102,:]
    test_data = df.iloc[102:, :]

    #build graph to get relationships for feature construction
    adj=build_graph()

    # Generate features for each column
    features = [getFeatures(col,adj) for col in range(df.shape[1])]
    print('number of features', len(features))

    #random search
    best_p={}
    # Fit VAR model for each column
    for col, feat in enumerate(features):

        min_rse=999999999
        for it in range(30):

            p=random.randint(1,10)
            print('experiment',e+1)
            print('column',col+1)
            print('iteration',it+1)

            # Fit VAR model with specified lag order and hyperparameters
            model = VAR(train_data[[col] + feat].values)
            model_fit = model.fit(maxlags=p)
                        
            # Forecast 36 months ahead
            forecast = model_fit.forecast(model_fit.endog,steps=36)
            forecast = forecast[:, 0]  # Extract only the forecast for the column itself
            valid= torch.tensor(valid_data[col].values)
            forecast = torch.tensor(forecast)
            rrse=calculate_rrse(valid,forecast)
            if rrse<min_rse:
                min_rse=rrse
                best_p[col]=p



    # Fit VAR model for each column
    forecast_results = {}
    final_train_data=df.iloc[:102, :] #merge train and valid
    for col, feat in enumerate(features):
        # Fit VAR model with specified lag order and hyperparameters
        model = VAR(final_train_data[[col] + feat].values)
        model_fit = model.fit(maxlags=best_p[col])
                        
        # Forecast 36 months ahead
        forecast = model_fit.forecast(model_fit.endog,steps=36)
        forecast_results[col] = forecast[:, 0]  # Extract only the forecast for the column itself


    # Convert forecast results to DataFrame
    forecast_df = pd.DataFrame(forecast_results)
    predict = torch.tensor(forecast_df.values)

    # Convert actual test data to PyTorch tensor
    test= torch.tensor(test_data.values)

    #check shapes
    print('test vs predict shapes',test.shape,predict.shape)

    rse=calculate_rrse(test,predict)
    rae=calculate_rae(test,predict)

    print('rrse=',rse)
    print('rae=',rae)
    print('best p',best_p)


    # Save forecast results to CSV
    forecast_df.to_csv('forecast_results_VAR.csv', index=False)

    return rse, rae

experiment=5
rse_l=[]
rae_l=[]
for e in range(experiment):
    rse, rae = main(e)
    rse_l.append(rse)
    rae_l.append(rae)

RSE= sum(rse_l)/len(rse_l)
RAE = sum (rae_l)/len(rae_l)

print('RSE=',RSE)
print('RAE=',RAE)