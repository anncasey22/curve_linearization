import torch
import torch.nn as nn
import numpy as np

class MLPBreakpointPredictor(nn.Module):
    def __init__(self, input_dim=100, hidden_dims=[64, 32, 16], output_dim=100):
        super(MLPBreakpointPredictor, self).__init__()
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

    def evaluate(self, input_signal, ground_truth):
        """
        input_signal: 1D numpy array
        ground_truth: 1D numpy array of same length
        Returns: (mse, segment_count)
        """
        self.eval()
        with torch.no_grad():
            input_tensor = torch.tensor(input_signal, dtype=torch.float32).unsqueeze(0)
            prediction = self.forward(input_tensor).squeeze(0).numpy()
            mse = np.mean((prediction - ground_truth) ** 2)
        return mse, 1  # MLP outputs a single smoothed segment
