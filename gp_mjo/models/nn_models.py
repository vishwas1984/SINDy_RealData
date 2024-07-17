import torch
import torch.nn as nn


class FFNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, seed=99):
        super(FFNNModel, self).__init__()
        self.seed = seed
        # Linear function
        self.fc1 = nn.Linear(input_dim, hidden_dim) 

        # Non-linearity activation function
        # self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

        # Linear function (readout)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

        
        self.dropout = nn.Dropout(p=0.25)

    def forward(self, x):

        # normalize x to [0,1]
        x -= x.min()
        x /= x.max()
        
        # Linear function  # LINEAR
        out = self.fc1(x)

        # Non-linearity  # NON-LINEAR
        out = self.relu(out)

        torch.manual_seed(self.seed)
        # Pass data through dropout
        out = self.dropout(out)

        # Linear function (readout)  # LINEAR
        out = self.fc2(out)
        return out