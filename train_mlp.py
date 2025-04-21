import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

from generate_curves import generate_sine_wave
from models.mlp_model import MLPBreakpointPredictor

#dataset
print("Generating training data...")
X = []  # noisy input
Y = []  # clean ground truth

for _ in range(5000):
    _, y_noisy, y_clean = generate_sine_wave(length=100, noise_level=0.05)
    X.append(y_noisy)
    Y.append(y_clean)

X_tensor = torch.tensor(X, dtype=torch.float32)
Y_tensor = torch.tensor(Y, dtype=torch.float32)

#dataloader
dataset = TensorDataset(X_tensor, Y_tensor)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

#model
model = MLPBreakpointPredictor(input_dim=100, output_dim=100)
optimizer = optim.Adam(model.parameters(), lr=0.001)

#training
n_epochs = 50
losses = []
print("Training model...")

for epoch in range(n_epochs):
    total_loss = 0.0
    for batch_x, batch_y in loader:
        optimizer.zero_grad()
        y_pred = model(batch_x)  # direct prediction from MLP
        loss = nn.MSELoss()(y_pred, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(loader)
    losses.append(avg_loss)
    print(f"Epoch {epoch+1}/{n_epochs}, Loss: {avg_loss:.4f}")

#save model
torch.save(model.state_dict(), "mlp_breakpoints.pt")
print("Model saved as mlp_breakpoints.pt")

#plot
plt.plot(losses)
plt.title("Training Loss Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.grid(True)
plt.show()
