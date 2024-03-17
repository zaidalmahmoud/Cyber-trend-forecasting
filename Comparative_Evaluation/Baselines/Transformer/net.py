import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import sys
import math

class TransformerModel(nn.Module):
    def __init__(self, input_size, output_size,nhead,num_encoder_layers,num_decoder_layers):
        super(TransformerModel, self).__init__()
        self.transformer = nn.Transformer(d_model=input_size, nhead=nhead, num_encoder_layers=num_encoder_layers, num_decoder_layers=num_decoder_layers)
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, input):
        transformer_out = self.transformer(input, input)  # Transformer requires same input for encoder and decoder
        output = self.fc(transformer_out)
        return output
