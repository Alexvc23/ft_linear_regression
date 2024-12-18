import json
import argparse
import numpy as np
from linear_regression.tools import read_csv, plot_normalized_regression, plot_cost_function, plot_original_data_with_regression, normalize_data

cost_history = []  # Track cost function over iterations

def compute_cost(theta0, theta1, x, y, m):
    # Compute the cost function, which measures the accuracy of the linear regression model
    cost = 0
    for i in range(m):
        predicted = theta0 + theta1 * x[i]  # Predicted value for the i-th data point
        cost += (predicted - y[i]) ** 2  # Squared error for the i-th data point
    return cost / (2 * m)  # Average squared error divided by 2

def train_model(x, y, learning_rate=0.001, max_iterations=1000, tolerance=1e-6, plot_debug=False):
    theta0 = 0  # Initialize theta0 (intercept)
    theta1 = 0  # Initialize theta1 (slope)
    m = len(x)  # Number of data points
    previous_cost = float('inf')  # Initialize previous cost to a large value
    current_tolerance = float('inf')  # Initialize tolerance value to 0

    for it in range(max_iterations):
        tmp_theta0 = 0
        tmp_theta1 = 0
        for i in range(m):
            predicted = theta0 + theta1 * x[i]  # Predicted value
            error = predicted - y[i]  # Error for the i-th data point
            tmp_theta0 += error  # Accumulate gradient for theta0
            tmp_theta1 += error * x[i]  # Accumulate gradient for theta1

        # Update theta values simultaneously
        theta0 -= (learning_rate / m) * tmp_theta0
        theta1 -= (learning_rate / m) * tmp_theta1

        # Compute the cost to check for convergence
        cost = compute_cost(theta0, theta1, x, y, m)
        cost_history.append(cost)

        if plot_debug and (it % 100 == 0 or it == max_iterations - 1):  # Conditional plotting for debugging
            plot_normalized_regression(x, y, theta0, theta1, "Training Progress", it, cost)
        
        current_tolerance = abs((previous_cost - cost) / previous_cost)  # Relative tolerance
        print(f"Iteration {it}: Cost = {cost:.6f}, Relative Tolerance = {current_tolerance:.6f}")
        if current_tolerance < tolerance:  # Check if the relative improvement is below the threshold
            print(f"Convergence reached at iteration {it}. Final Cost = {cost:.6f}")
            break

        previous_cost = cost  # Update the previous cost

    return theta0, theta1

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Linear Regression Training")
    parser.add_argument("file_path", type=str, help="Path to the CSV file containing data")
    parser.add_argument("--plot_debug", action="store_true", help="Enable debug plotting during training")
    args = parser.parse_args()

    print("Reading data from CSV...")
    try:
        x, y = read_csv(args.file_path)  # Load data from the CSV file
        x = np.array(x)  # Convert input data to a NumPy array
        y = np.array(y)  # Convert target data to a NumPy array

        original_x = x.copy()  # Save original input data for plotting
        original_y = y.copy()  # Save original target data for denormalization and plotting

        x_mean, x_std = np.mean(x), np.std(x)
        y_mean, y_std = np.mean(y), np.std(y)

        x = normalize_data(x)  # Normalize input data
        y = normalize_data(y)  # Normalize target data

    except ValueError as e:
        print(f"Error: {e}")
        return

    learning_rate = 0.01  # Learning rate for gradient descent
    max_iterations = 100000  # Maximum number of iterations for training
    tolerance = 1e-6  # Convergence threshold: the minimum change in cost function to continue iterations

    print("Training the model...")
    theta0, theta1 = train_model(x, y, learning_rate, max_iterations, tolerance, plot_debug=args.plot_debug)  # Train the model
    print(f"Training complete. Theta0 = {theta0:.6f}, Theta1 = {theta1:.6f}")

    plot_original_data_with_regression(original_x, original_y, theta0, theta1, x_mean, x_std, y_mean, y_std)  # Plot non-normalized data with regression line
    plot_cost_function(cost_history)  # Plot the cost function over iterations

    mean_price = np.mean(original_y)  # Mean of the original target data
    std_price = np.std(original_y)  # Standard deviation of the original target data

    # Save the model parameters and normalization information to a JSON file
    with open("theta_values.json", "w") as file:
        json.dump({"theta0": theta0, "theta1": theta1, "mean": np.mean(original_x), "std": np.std(original_x), "mean_price": mean_price, "std_price": std_price}, file)
    print("Theta values and scaling parameters saved for prediction program.")

if __name__ == "__main__":
    main()
