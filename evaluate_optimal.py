import numpy as np
import torch
import matplotlib.pyplot as plt

from models.mlp_model import MLPBreakpointPredictor
from algorithms.linearize_greedy import linearize_curve
from algorithms.optimize_lines import optimal_segment_fit, compute_segment_err


class CurveLinearizationComparison:
    def __init__(self):
        self.results = {}
        self.model = None

    def load_nn_model(self, model_path="mlp_breakpoints.pt", input_dim=100, hidden_dims=[64, 32, 16], output_dim=100):
        """Load the trained neural network model"""
        self.model = MLPBreakpointPredictor(input_dim, hidden_dims, output_dim)
        try:
            self.model.load_state_dict(torch.load(model_path))
            self.model.eval()
            print(f"Model loaded from {model_path}")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Neural network method will not be available")
            self.model = None
            return False

    def predict_nn_breakpoints(self, x_values, y_values, threshold=0.1):
        """
        - convert neural network output to breakpoints
       - MLP doesn't directly output breakpoints
        """
        if self.model is None:
            return []

        with torch.no_grad():
            input_tensor = torch.tensor(y_values, dtype=torch.float32).unsqueeze(0)
            prediction = self.model.forward(input_tensor).squeeze(0).numpy()

            diffs = np.abs(np.diff(prediction))
            threshold_value = np.mean(diffs) + threshold * np.std(diffs)

            #potential breakpoints
            breakpoints = [0]  # Always include first point
            for i in range(len(diffs)):
                if diffs[i] > threshold_value:
                    breakpoints.append(i + 1)

            #last point
            if len(y_values) - 1 not in breakpoints:
                breakpoints.append(len(y_values) - 1)

            #result with proper (x,y) coordinates
            result = [(x_values[i], y_values[i]) for i in breakpoints]
            return result

    def preprocess_data(self, x_values, y_values):
        """Convert separate x and y arrays to list of (x,y) points"""
        return list(zip(x_values, y_values))

    def compare_methods(self, x_values, y_values, ground_truth=None, epsilon=0.01, max_error=0.01, threshold=0.1):
        """
        - compare all methods on the same dataset
        """
        points = self.preprocess_data(x_values, y_values)
        self.results = {}

        #greedy
        greedy_points = linearize_curve(points, epsilon=epsilon)
        greedy_mse = self.calculate_mse(self.preprocess_data(x_values, ground_truth), greedy_points)
        self.results["Greedy"] = {
            "linearized_points": greedy_points,
            "segment_count": len(greedy_points) - 1,
            "mse": greedy_mse,
            "gt_mse": None
        }

        #dp
        dp_points = optimal_segment_fit(points, max_error=max_error)
        dp_mse = self.calculate_mse(points, dp_points)
        self.results["Dynamic Programming"] = {
            "linearized_points": dp_points,
            "segment_count": len(dp_points) - 1,
            "mse": dp_mse,
            "gt_mse": None
        }

        #nueral network
        if self.model is not None:
            # Get breakpoints from neural network prediction
            nn_points = self.predict_nn_breakpoints(x_values, y_values, threshold)
            nn_mse = self.calculate_mse(points, nn_points)

            if ground_truth is not None:
                model_mse, _ = self.model.evaluate(y_values, ground_truth)
            else:
                model_mse = None

            self.results["Neural Network"] = {
                "linearized_points": nn_points,
                "segment_count": len(nn_points) - 1,
                "mse": nn_mse,
                "gt_mse": model_mse,
                "model_prediction": self.model(torch.tensor(y_values, dtype=torch.float32).unsqueeze(0)).squeeze(0).detach().numpy()
            }

        #ground truth MSE if provided
        if ground_truth is not None:
            gt_points = self.preprocess_data(x_values, ground_truth)
            for method in self.results:
                linearized_points = self.results[method]["linearized_points"]
                gt_mse = self.calculate_mse(gt_points, linearized_points)
                self.results[method]["gt_mse"] = gt_mse

        return self.results

    def calculate_mse(self, original_points, linearized_points):
        """
        -mean squared error between original curve and linearized representation
        """
        if len(linearized_points) <= 1:
            return float('inf')

        # each segment in linearized points, calculate error for points in that range
        total_squared_error = 0
        total_points = 0

        original_x = [p[0] for p in original_points]
        min_x, max_x = min(original_x), max(original_x)

        for i in range(len(linearized_points) - 1):
            # Get segment endpoints
            x1, y1 = linearized_points[i]
            x2, y2 = linearized_points[i + 1]

            # segment equation
            if x2 - x1 == 0:  # Vertical line
                continue

            slope = (y2 - y1) / (x2 - x1)
            intercept = y1 - slope * x1

            #original points that fall within this segment's x range
            segment_points = [p for p in original_points if x1 <= p[0] <= x2]

            # error for each point
            for x, y in segment_points:
                y_pred = slope * x + intercept
                total_squared_error += (y - y_pred) ** 2
                total_points += 1

        # mean squared error
        return total_squared_error / total_points if total_points > 0 else float('inf')

    def plot_comparison(self, x_values, y_values, ground_truth=None, title="Curve Linearization Methods Comparison"):
        """
        Plot the original curve and all linearization results
        """
        if not self.results:
            print("No results to plot. Run compare_methods first.")
            return

        plt.figure(figsize=(12, 8))

        plt.plot(x_values, y_values, 'k-', alpha=0.5, label='Original Data')

        # Plot ground truth if provided
        if ground_truth is not None:
            plt.plot(x_values, ground_truth, 'b--', alpha=0.7, label='Ground Truth')

        colors = ['r', 'g', 'm', 'c', 'y']
        for i, (name, result) in enumerate(self.results.items()):
            linearized_points = result["linearized_points"]
            x_linear = [p[0] for p in linearized_points]
            y_linear = [p[1] for p in linearized_points]

            color = colors[i % len(colors)]
            plt.plot(x_linear, y_linear, f'{color}o-', linewidth=2,
                     label=f'{name} (segments={result["segment_count"]}, MSE={result["mse"]:.6f})')

            if name == "Neural Network" and "model_prediction" in result:
                plt.plot(x_values, result["model_prediction"], 'c:', alpha=0.7,
                         label="NN Raw Prediction")

        plt.title(title)
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        plt.figure(figsize=(10, 6))
        method_names = list(self.results.keys())
        mse_values = [self.results[name]["mse"] for name in method_names]
        segment_counts = [self.results[name]["segment_count"] for name in method_names]

        #bar chart for MSE
        plt.subplot(1, 2, 1)
        plt.bar(method_names, mse_values, color=['r', 'g', 'm'])
        plt.title("Mean Squared Error Comparison")
        plt.ylabel("MSE")
        plt.xticks(rotation=45)

        #bar chart for segment counts
        plt.subplot(1, 2, 2)
        plt.bar(method_names, segment_counts, color=['r', 'g', 'm'])
        plt.title("Segment Count Comparison")
        plt.ylabel("Number of Segments")
        plt.xticks(rotation=45)

        plt.tight_layout()
        plt.show()

    def print_summary(self):
        """Print summary of comparison results"""
        if not self.results:
            print("No results to summarize. Run compare_methods first.")
            return

        print("\n Method Comparison Summary ")
        print(f"{'Method':<20} {'Segments':<10} {'MSE':<15} {'GT MSE':<15}")
        for name, result in self.results.items():
            gt_mse_str = f"{result['gt_mse']:.6f}" if result['gt_mse'] is not None else "N/A"
            print(f"{name:<20} {result['segment_count']:<10} {result['mse']:.6f} {gt_mse_str}")




def main():
    #generate or load test data
    def generate_sample_curve(n_points=100, noise_level=0.05):
        x = np.linspace(0, 10, n_points)
        y_clean = np.sin(x) + 0.5 * np.sin(2 * x) + 0.2 * np.sin(5 * x)
        y_noisy = y_clean + noise_level * np.random.randn(n_points)
        return x, y_noisy, y_clean

    #test data
    x, y_noisy, y_clean = generate_sample_curve(100, 0.1)

    comparison = CurveLinearizationComparison()

    #try to load neural network model
    comparison.load_nn_model("mlp_breakpoints.pt")

    #compare methods
    results = comparison.compare_methods(
        x_values=x,
        y_values=y_noisy,
        ground_truth=y_clean,
        epsilon=0.05,
        max_error=0.05
    )

    comparison.print_summary()

    #plot
    comparison.plot_comparison(x, y_noisy, y_clean, "Curve Linearization Methods Comparison")


if __name__ == "__main__":
    main()