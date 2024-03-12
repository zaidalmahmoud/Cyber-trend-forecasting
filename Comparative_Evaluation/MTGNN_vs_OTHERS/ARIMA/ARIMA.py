import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import torch
import sys
import math

# Read the dataset from CSV file
df = pd.read_csv('sm_data.csv', header=None)

# Take the first 102 rows as training data and the rest as test data
train_data = df.iloc[:102, :]
test_data = df.iloc[102:, :]


# Iterate over each trend
forecast_results = {}
for column in df.columns:
    # Fit ARIMA model on training data
    model = ARIMA(train_data[column], order=(10,1,0))
    model_fit = model.fit()


    # Forecast 36 months ahead
    forecast = model_fit.forecast(steps=36)

    # Store forecast results
    forecast_results[column] = forecast

# Convert forecast results to DataFrame
forecast_df = pd.DataFrame(forecast_results)

# Save forecast results to CSV
forecast_df.to_csv('forecast_results_ARIMA.csv', index=False)

# Convert actual test data to PyTorch tensor
test = torch.tensor(test_data.values)

# Convert forecast data to PyTorch tensor
predict = torch.tensor(forecast_df.values)



# Print the size of the forecast tensor
print("Size of predict tensor:", predict.size())
print("Size of test tensor:", test.size())




#testing:

#RRSE according to Lai et.al
sum_squared_diff = torch.sum(torch.pow(test - predict, 2))
#Relative Absolute Error RAE 
sum_absolute_diff= torch.sum(torch.abs(test - predict))


##########################################################################################################
#RRSE according to Lai et.al
root_sum_squared= math.sqrt(sum_squared_diff) #numerator

# scale = data.scale.expand(test.size(0), data.m) #scale will have the max of each column (142 max values)
# test_s= test*scale
test_s=test
mean_all = torch.mean(test_s, dim=0) # calculate the mean of each column in test call it Yj-
diff_r = test_s - mean_all.expand(test_s.size(0), 142) # subtract the mean from each element in the tensor test
sum_squared_r = torch.sum(torch.pow(diff_r, 2))# square the result and sum over all elements
root_sum_squared_r=math.sqrt(sum_squared_r)#denominator

#RRSE according to Lai et.al
rrse=root_sum_squared/root_sum_squared_r
print('rrse=',rrse)
###########################################################################################################
###########################################################################################################

#Relative Absolute Error RAE 
sum_absolute_r=torch.sum(torch.abs(diff_r))# absolute the result and sum over all elements
rae=sum_absolute_diff/sum_absolute_r 
rae=rae.item()
print('rae=',rae)

###########################################################################################################

