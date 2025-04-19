
import numpy as np
from utils.linearize_greedy import linearize_curve

def generate_sine_wave(length=100, freq_range=(1, 3), noise_level=0.05):
    """
    Generate sine wave with random frequency and optional noise.
    """
    freq = np.random.uniform(*freq_range)
    x = np.linspace(0, 2 * np.pi, length)
    y = np.sin(freq * x) + np.random.normal(0, noise_level, size=length)
    return y

def generate_curve_dataset(n_samples=1000, length=100, epsilon=0.01):
    """
    Generate a dataset of (y_values, breakpoint_labels) for training the CNN.
    """
    X = []
    Y = []

    for _ in range(n_samples):
        y = generate_sine_wave(length)
        x_vals = np.linspace(0, 2 * np.pi, length)
        points = list(zip(x_vals, y))
        segments = linearize_curve(points, epsilon)

        # Label breakpoints: 1 if it's a segment endpoint, else 0
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
