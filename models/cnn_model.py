# models/nn_model.py

import torch
import torch.nn as nn

class CNNLinearizer(nn.Module):
    def __init__(self, seq_len):
        super(CNNLinearizer, self).__init__()
        self.seq_len = seq_len
        self.net = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=5, padding=2),  # keeps same size
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(32, 1, kernel_size=1),  # output 1 value per position
            nn.Sigmoid()  # outputs between 0 and 1
        )

    def forward(self, x):
        return self.net(x).squeeze(1)  # shape: [batch_size, seq_len]
