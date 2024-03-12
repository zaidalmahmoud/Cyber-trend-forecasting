import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import sys
import math

class LSTMModel2(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(LSTMModel2, self).__init__()
        self.num_layers = len(hidden_sizes)
        self.lstm_layers = nn.ModuleList([
            nn.LSTM(input_size if i == 0 else hidden_sizes[i-1], hidden_sizes[i])
            for i in range(self.num_layers)
        ])
        self.fc = nn.Linear(hidden_sizes[-1], output_size)

    def forward(self, input):
        # Forward pass through each LSTM layer
        for lstm_layer in self.lstm_layers:
            input, _ = lstm_layer(input)

        # Final output from the last LSTM layer
        output = self.fc(input)
        return output

# Read the dataset from CSV file
df = pd.read_csv('sm_data.csv', header=None)

# Take the first 92 rows as training data and the rest as test data
train_data = df.iloc[:102, :]
test_data = df.iloc[92:, :]

# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
train_data_normalized = scaler.fit_transform(train_data)
test_data_normalized = scaler.transform(test_data)

# Convert training data to PyTorch tensor and reshape for input
train_tensor_normalized = torch.tensor(train_data_normalized, dtype=torch.float32)
train_tensor_normalized = train_tensor_normalized.reshape(-1, 142)
print('train tensor shape:', train_tensor_normalized.shape)

# Define window size
input_window = 10
output_window = 36

# Create batches of input-output pairs using sliding window approach
train_batches = []
for i in range(len(train_tensor_normalized) - input_window - output_window + 1):
    input_seq = train_tensor_normalized[i:i + input_window]
    output_seq = train_tensor_normalized[i + input_window:i + input_window + output_window]
    train_batches.append((input_seq, output_seq))

# Print the number of batches
print("Number of training batches:", len(train_batches))

# Define hyperparameters
input_size = 10
hidden_size = [64, 32]
output_size = 36
num_epochs = 100
learning_rate = 0.001

# Initialize the model, loss function, and optimizer
forecast = {}
for col in df.columns:
    print('************training column', col, '***********************')
    model = LSTMModel2(input_size, hidden_size, output_size)
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # Train the LSTM model
    for epoch in range(num_epochs):
        total_loss = 0
        for input_seq, target_seq in train_batches:
            optimizer.zero_grad()
            outputs = model(input_seq[:, col].unsqueeze(0)).squeeze(0)
            loss = criterion(outputs, target_seq[:, col])  # Compare last output with ground truth
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if (epoch + 1) % 10 == 0:
            print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, total_loss / len(train_batches)))
            print(outputs)

    test_tensor_normalized_col = torch.tensor(test_data_normalized[:10, col], dtype=torch.float32).unsqueeze(0)
    forecast[col] = model(test_tensor_normalized_col).squeeze(0)
    print(forecast[col].shape)

dict_values = list(forecast.values())
numpy_arrays =  [tensor.detach().numpy() for tensor in dict_values]
# Convert the values to a NumPy array
forecast = np.transpose(np.array(numpy_arrays))
print('forecast shape', forecast.shape)

# Inverse normalization on the forecasted values
forecast_denormalized= scaler.inverse_transform(forecast)
print('forecast normalised shape' , forecast_denormalized.shape)
# Convert forecast results to DataFrame
forecast_df = pd.DataFrame(forecast_denormalized)

# Save forecast results to CSV
forecast_df.to_csv('forecast_results_normalised_u.csv', index=False)

# Inverse normalization on the test data (if needed)
test_denormalized = scaler.inverse_transform(test_data_normalized)

# Convert the forecast DataFrame to PyTorch tensor
predict = torch.tensor(forecast_df.values)

# Convert the test data to PyTorch tensor
test = torch.tensor(test_denormalized)

# Extract the test data after the initial 10 rows
test = test[10:]


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