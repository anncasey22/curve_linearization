# evaluate_optimality.py

import torch
import numpy as np
from sklearn.metrics import f1_score
from models.cnn_model import CNNLinearizer
from utils.linearize_greedy import linearize_curve
from utils.optimize_lines import optimal_segment_fit
from generate_curves import generate_sine_wave


def compute_mse(original, approx):
    x_orig, y_orig = zip(*original)
    x_approx, y_approx = zip(*approx)
    y_interp = np.interp(x_orig, x_approx, y_approx)
    return np.mean((np.array(y_orig) - y_interp) ** 2)


def breakpoint_vector(points, breakpoints):
    x_vals = [p[0] for p in points]
    vec = np.zeros(len(points))
    for pt in breakpoints:
        idx = np.argmin(np.abs(np.array(x_vals) - pt[0]))
        vec[idx] = 1
    return vec


# ==== Config ====
seq_len = 100
n_tests = 50
threshold = 0.3

# ==== Load model ====
model = CNNLinearizer(seq_len=seq_len)
model.load_state_dict(torch.load("cnn_linearizer.pt"))
model.eval()

# ==== Run evaluation ====
total_results = []

for _ in range(n_tests):
    y = generate_sine_wave(length=seq_len, noise_level=0.05)
    x = np.linspace(0, 2 * np.pi, seq_len)
    points = list(zip(x, y))

    # CNN prediction
    with torch.no_grad():
        input_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        preds = model(input_tensor).squeeze().numpy()
    break_indices = np.where(preds > threshold)[0]
    cnn_segments = [points[0]] + [points[i] for i in break_indices] + [points[-1]]

    # Baseline & DP
    greedy_segments = linearize_curve(points, epsilon=0.01)
    dp_segments = optimal_segment_fit(points, max_error=0.01)

    # Metrics
    mse_cnn = compute_mse(points, cnn_segments)
    mse_dp = compute_mse(points, dp_segments)
    mse_greedy = compute_mse(points, greedy_segments)

    # F1 Score for CNN vs Greedy (as label reference)
    y_true = breakpoint_vector(points, greedy_segments)
    y_pred = breakpoint_vector(points, cnn_segments)
    f1 = f1_score(y_true, y_pred, zero_division=1)

    total_results.append({
        'mse_cnn': mse_cnn,
        'mse_dp': mse_dp,
        'mse_greedy': mse_greedy,
        'segments_cnn': len(cnn_segments),
        'segments_dp': len(dp_segments),
        'segments_greedy': len(greedy_segments),
        'f1_score': f1
    })

# ==== Summary ====
mse_cnn_avg = np.mean([r['mse_cnn'] for r in total_results])
mse_dp_avg = np.mean([r['mse_dp'] for r in total_results])
mse_greedy_avg = np.mean([r['mse_greedy'] for r in total_results])

segments_cnn_avg = np.mean([r['segments_cnn'] for r in total_results])
segments_dp_avg = np.mean([r['segments_dp'] for r in total_results])
segments_greedy_avg = np.mean([r['segments_greedy'] for r in total_results])

f1_avg = np.mean([r['f1_score'] for r in total_results])

print("\n===== EVALUATION SUMMARY ({} test cases) =====".format(n_tests))
print(f"Avg MSE:     CNN={mse_cnn_avg:.4f}, DP={mse_dp_avg:.4f}, Greedy={mse_greedy_avg:.4f}")
print(f"Avg Segments: CNN={segments_cnn_avg:.2f}, DP={segments_dp_avg:.2f}, Greedy={segments_greedy_avg:.2f}")
print(f"Avg F1 vs Greedy (CNN): {f1_avg:.2f}")
print("================================================\n")
