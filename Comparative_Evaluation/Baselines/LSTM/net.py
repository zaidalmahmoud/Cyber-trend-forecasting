import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import sys
import math

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(LSTMModel, self).__init__()
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
