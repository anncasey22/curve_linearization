
import torch
import numpy as np
import matplotlib.pyplot as plt

from models.cnn_model import CNNLinearizer
from utils.linearize_greedy import linearize_curve
from utils.optimize_lines import optimal_segment_fit
from generate_curves import generate_sine_wave

#load trained model
seq_len = 100
model = CNNLinearizer(seq_len=seq_len)
model.load_state_dict(torch.load("cnn_linearizer.pt"))
model.eval()

#generate a new test curve
y = generate_sine_wave(length=seq_len, noise_level=0.05)
x = np.linspace(0, 2 * np.pi, seq_len)
points = list(zip(x, y))

#get CNN predictions
with torch.no_grad():
    input_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len]
    preds = model(input_tensor).squeeze().numpy()  # shape: [seq_len]

#threshold predictions to find breakpoints
threshold = 0.5
break_indices = np.where(preds > threshold)[0]
cnn_segments = [points[0]] + [points[i] for i in break_indices] + [points[-1]]

#greedy and DP segments
greedy_segments = linearize_curve(points, epsilon=0.01)
dp_segments = optimal_segment_fit(points, max_error=0.01)

#plot everything
def plot_segments(segments, label, marker, style):
    xs, ys = zip(*segments)
    plt.plot(xs, ys, marker+style, label=label)

plt.figure(figsize=(10, 6))
plt.plot(x, y, label="Original Curve", alpha=0.4)
plot_segments(greedy_segments, "Greedy", 'o', '-')
plot_segments(dp_segments, "DP Optimized", 'x', '--')
plot_segments(cnn_segments, "CNN Prediction", '*', ':')

plt.legend()
plt.title("Segment Comparison: Greedy vs DP vs CNN")
plt.xlabel("x")
plt.ylabel("y")
plt.grid(True)
plt.show()