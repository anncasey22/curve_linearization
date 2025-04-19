
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

from models.cnn_model import CNNLinearizer
from generate_curves import generate_curve_dataset

torch.manual_seed(0)

#generate dataset
print("Generating training data...")
X, Y = generate_curve_dataset(n_samples=5000, length=100, epsilon=0.01)

#convert to pytorch tensors
X_tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(1)  # [batch, 1, seq_len]
Y_tensor = torch.tensor(Y, dtype=torch.float32)               # [batch, seq_len]

#dataloader
dataset = TensorDataset(X_tensor, Y_tensor)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

#define the model
model = CNNLinearizer(seq_len=100)
criterion = nn.BCELoss()  # Binary cross-entropy for segmentation
optimizer = optim.Adam(model.parameters(), lr=0.001)

#train the model
n_epochs = 50
losses = []

print("Training model...")
for epoch in range(n_epochs):
    total_loss = 0.0
    for batch_x, batch_y in loader:
        optimizer.zero_grad()
        outputs = model(batch_x)  # shape: [batch, seq_len]
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(loader)
    losses.append(avg_loss)
    print(f"Epoch {epoch+1}/{n_epochs}, Loss: {avg_loss:.4f}")

#save the model
torch.save(model.state_dict(), "cnn_linearizer.pt")
print("Model saved as cnn_linearizer.pt")

#plot loss curve
plt.plot(losses)
plt.title("Training Loss Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Binary Cross-Entropy Loss")
plt.grid(True)
plt.show()
