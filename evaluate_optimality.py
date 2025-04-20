import torch
import numpy as np
from models.mlp_model import MLPBreakpointPredictor
from utils.optimize_lines import optimal_segment_fit
from generate_curves import generate_sine_wave

# ==== Helper: Compute MSE ====
def compute_mse(y_true, y_pred):
    return np.mean((np.array(y_true) - np.array(y_pred)) ** 2)

# ==== DP Class ====
class DPMethod:
    def __init__(self, epsilon=0.01):
        self.epsilon = epsilon

    def evaluate(self, x, y, points):
        dp_pts = optimal_segment_fit(points, max_error=self.epsilon)
        y_dp = np.interp(x, [pt[0] for pt in dp_pts], [pt[1] for pt in dp_pts])
        mse = compute_mse(y, y_dp)
        return mse, len(dp_pts)

# ==== MLP Class ====
class MLPMethod:
    def __init__(self, model_path="mlp_breakpoints.pt"):
        self.model = MLPBreakpointPredictor(input_dim=100, output_dim=100)
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()

    def evaluate(self, y):
        with torch.no_grad():
            input_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(0)
            y_pred = self.model(input_tensor).squeeze(0).numpy()
        mse = compute_mse(y, y_pred)
        return mse, 1  # 1 segment since it's a smoothed prediction

# ==== Config ====
seq_len = 100
n_tests = 50
noise = 0.05

# ==== Initialize Methods ====
dp = DPMethod(epsilon=0.01)
mlp = MLPMethod()

# ==== Store results ====
mse_dp = []
segments_dp = []
mse_mlp = []
segments_mlp = []

# ==== Run comparisons ====
for _ in range(n_tests):
    y = generate_sine_wave(length=seq_len, noise_level=noise)
    x = np.linspace(0, 2 * np.pi, seq_len)
    points = list(zip(x, y))

    dp_mse, dp_segs = dp.evaluate(x, y, points)
    mlp_mse, mlp_segs = mlp.evaluate(y)

    mse_dp.append(dp_mse)
    segments_dp.append(dp_segs)
    mse_mlp.append(mlp_mse)
    segments_mlp.append(mlp_segs)

# ==== Report Summary ====
print("\n===== AVERAGE RESULTS OVER {} TEST CASES =====".format(n_tests))
print(f"DP     MSE: {np.mean(mse_dp):.4f}, Avg Segments: {np.mean(segments_dp):.2f}")
print(f"MLP    MSE: {np.mean(mse_mlp):.4f}, Avg Segments: {np.mean(segments_mlp):.2f}")
print("===============================================\n")
