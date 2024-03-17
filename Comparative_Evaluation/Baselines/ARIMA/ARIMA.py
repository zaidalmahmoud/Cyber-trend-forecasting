import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import torch
import sys
import math
import random
import numpy as np


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

def main(e):
    # Read the dataset from CSV file
    df = pd.read_csv('sm_data.csv', header=None)

    # Take the first 102 rows as training data and the rest as test data
    train_data = df.iloc[:66, :]
    valid_data = df.iloc[66:102,:]
    test_data = df.iloc[102:, :]
 

    # Iterate over each trend
    best_order={}
    for column in df.columns:
       #random search for this particular column
        
        min_rse=999999999    
        for it in range(30):
            p = 10
            d = random.randint(0, 2) 
            q = random.randint(0, 10)

            print('experiment',e+1)
            print('column',column+1)
            print('iteration',it+1)

            # Fit ARIMA model on training data
            model = ARIMA(train_data[column], order=(p,d,q))
            try:
                model_fit = model.fit()
            except:
                for _ in range(5):
                    print('ERRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRROR')
                d=1
                q=0
                model = ARIMA(train_data[column], order=(p,d,q))
                model_fit = model.fit()

            # Forecast 36 months ahead
            forecast = model_fit.forecast(steps=36)
            valid= torch.tensor(valid_data[column].values)
            forecast = torch.tensor(forecast.values, dtype=valid.dtype)
            rrse=calculate_rrse(valid,forecast)
            if rrse<min_rse:
                min_rse=rrse
                best_order[column]=(p,d,q)



    #best model of the 30 is ready now:
    # Iterate over each trend
    forecast_results = {}
    final_train_data=df.iloc[:102, :] #merge train and valid
    for column in df.columns:
        # Fit ARIMA model on training data
        model = ARIMA(final_train_data[column], order=best_order[column])
        try:
            model_fit = model.fit()
        except:
            for _ in range(5):
                print('ERRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRROR')
            sys.exit()

        # Forecast 36 months ahead
        forecast = model_fit.forecast(steps=36)

        # Store forecast results
        forecast_results[column] = forecast

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


    # Save forecast results to CSV
    forecast_df.to_csv('forecast_results_ARIMA.csv', index=False)

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