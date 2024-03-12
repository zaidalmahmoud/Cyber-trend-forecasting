import pandas as pd
from statsmodels.tsa.api import VAR
import torch
import sys
import math

def getFeatures(col):
    f=[]
    if col < 16:  # If column is less than 16 (attack column), append mention to the attack (NoM)
        f.append(col + 16)
    else:  # Otherwise, append the column itself
        f.append(col)
    
    f.append(42)  # Append the feature for wars (ACA)
    f.append(43)  # Append the feature for holidays (PH)
    return f

def getALLFeatures(col,m):
    return  [i for i in range(m) if i != col]



# Read the dataset from CSV file
df = pd.read_csv('sm_data.csv', header=None)

# Take the first 102 rows as training data and the rest as test data
train_data = df.iloc[:102, :]
test_data = df.iloc[102:, :]

# Generate features for each column
features = [getFeatures(col) for col in range(df.shape[1])]
print(len(features))

# Fit separate VAR models for each column
models = {}
for col, feat in enumerate(features):
    model = VAR(train_data[[col] + feat].values)
    model_fit = model.fit()
    models[col] = model_fit

# Forecast 36 months ahead for each column
forecast_results = {}
for col, model_fit in models.items():
    forecast = model_fit.forecast(model_fit.endog,steps=36)
    forecast_results[col] = forecast[:, 0]  # Extract only the forecast for the column itself

# Convert forecast results to DataFrame
forecast_df = pd.DataFrame(forecast_results, columns=df.columns)

# Save forecast results to CSV
forecast_df.to_csv('forecast_results_VAR.csv', index=False)

# Convert actual test data to PyTorch tensor
test = torch.tensor(test_data.values)

# Convert forecast data to PyTorch tensor
predict = torch.tensor(forecast_df.values)

# Print the size of the forecast tensor
print("Size of predict tensor:", predict.size())
print("Size of test tensor:", test.size())

# Testing:

# RRSE according to Lai et.al
sum_squared_diff = torch.sum(torch.pow(test - predict, 2))

# Relative Absolute Error RAE
sum_absolute_diff = torch.sum(torch.abs(test - predict))

# RRSE according to Lai et.al
root_sum_squared = math.sqrt(sum_squared_diff)  # numerator
mean_all = torch.mean(test, dim=0)  # calculate the mean of each column in test
diff_r = test - mean_all.expand(test.size(0), test.size(1))  # subtract the mean from each element in the tensor test
sum_squared_r = torch.sum(torch.pow(diff_r, 2))  # square the result and sum over all elements
root_sum_squared_r = math.sqrt(sum_squared_r)  # denominator
rrse = root_sum_squared / root_sum_squared_r
print('rrse=', rrse)

# Relative Absolute Error RAE
sum_absolute_r = torch.sum(torch.abs(diff_r))  # absolute the result and sum over all elements
rae = sum_absolute_diff / sum_absolute_r
rae = rae.item()
print('rae=', rae)
