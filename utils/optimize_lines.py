import numpy as np

def compute_segment_err(points, i, j):
    """
    fit a line to points[i:j+1] and return the total squared error
    - try every valid segment using least squares fitting
    - use DP to find the fewest segments with error <= max_error
    - d[i] keeps track of the min number of segments up to i
    """
    segment = points[i:j + 1]
    x = np.array([p[0] for p in segment])
    y = np.array([p[1] for p in segment])

    A = np.vstack([x, np.ones(len(x))]).T
    slope, intercept = np.linalg.lstsq(A, y, rcond=None)[0]
    y_pred = slope * x + intercept
    return np.sum((y - y_pred) ** 2)


def optimal_segment_fit(points, max_error=0.01):
    """
    Use dynamic programming to find the minimum number of linear segments
    that approximate the curve within max_error per segment.
    """
    n = len(points)
    dp = [float('inf')] * n  #dp[i] = min segments to reach point i
    prev = [-1] * n  #previous split point
    dp[0] = 0

    for i in range(1, n):
        for j in range(i):
            err = compute_segment_err(points, j, i)
            if err <= max_error:
                if dp[j] + 1 < dp[i]:
                    dp[i] = dp[j] + 1
                    prev[i] = j

    #reconstruct segment points
    result = []
    idx = n - 1
    while idx >= 0:
        result.append(points[idx])
        idx = prev[idx]
    result.append(points[0])
    return list(reversed(result))


