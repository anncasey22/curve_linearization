
import numpy as np
from algorithms.linearize_greedy import linearize_curve

def generate_sine_wave(length=100, freq_range=(1, 3), noise_level=0.05):
    """
    -generate sine wave with random frequency and optional noise.
    -return
        x: np.array of shape [length]
        y_noisy: np.array of shape [length]
        y_clean: np.array of shape [length] (true sine)
    """
    freq = np.random.uniform(*freq_range)
    x = np.linspace(0, 2 * np.pi, length)
    y_clean = np.sin(freq * x)
    y_noisy = y_clean + np.random.normal(0, noise_level, size=length)
    return x, y_noisy, y_clean  # ✅ must return 3 things!

def generate_curve_dataset(n_samples=1000, length=100, epsilon=0.02):
    """
    -generate a dataset of (y_values, breakpoint_labels) for training the CNN.
    """
    X = []
    Y = []

    for _ in range(n_samples):
        y = generate_sine_wave(length)
        x_vals = np.linspace(0, 2 * np.pi, length)
        points = list(zip(x_vals, y))
        segments = linearize_curve(points, epsilon)

        # breakpoints: 1 if it's a segment endpoint, else 0
        segment_indices = set()
        for pt in segments:
            idx = np.argmin(np.abs(x_vals - pt[0]))
            segment_indices.add(idx)

        labels = np.zeros(length)
        for idx in segment_indices:
            labels[idx] = 1

        X.append(y)
        Y.append(labels)

    return np.array(X), np.array(Y)
