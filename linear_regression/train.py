import json
import numpy as np
import matplotlib.pyplot as plt
from linear_regression.tools import read_csv, plot_data_and_model, plot_data_only


def compute_cost(theta0, theta1, x, y, m):
    # Compute the cost function, which measures the accuracy of the linear regression model
    cost = 0
    for i in range(m):
        predicted = theta0 + theta1 * x[i]  # Predicted value for the i-th data point
        cost += (predicted - y[i]) ** 2  # Squared error for the i-th data point
    return cost / (2 * m)  # Average squared error divided by 2

def normalize_data(data):
    # Normalize data to have mean 0 and standard deviation 1
    return (data - np.mean(data)) / np.std(data)


def train_model(x, y, learning_rate=0.001, max_iterations=1000, tolerance=1e-6):
    theta0 = 0  # Initialize theta0 (intercept)
    theta1 = 0  # Initialize theta1 (slope)
    m = len(x)  # Number of data points
    previous_cost = float('inf')  # Initialize previous cost to a large value

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

        if it % 100 == 0 or it == max_iterations - 1:  # Log every 100 iterations and the last iteration
            plot_iteration(x, y, theta0, theta1, it, cost)
            current_tolerance = abs(previous_cost - cost)
            print(f"Iteration {it}: Cost = {cost:.6f}, Tolerance = {current_tolerance:.6f}")
        if abs(current_tolerance) < tolerance:  # Check if the improvement is below the threshold
            print(f"Convergence reached at iteration {it}. Final Cost = {cost:.6f}")
            break

        previous_cost = cost  # Update the previous cost

    return theta0, theta1

def main():
    # Main function to execute the linear regression training pipeline
    file_path = "/Users/alex/Documents/programing/42/projects/ft_linear_regression/dev/assets/data.csv"  # Replace with your actual file path

    print("Reading data from CSV...")
    try:
        x, y = read_csv(file_path)  # Load data from the CSV file
        x = np.array(x)  # Convert input data to a NumPy array
        y = np.array(y)  # Convert target data to a NumPy array

        original_x = x.copy()  # Save original input data for plotting
        original_y = y.copy()  # Save original target data for denormalization and plotting

        x = normalize_data(x)  # Normalize input data
        y = normalize_data(y)  # Normalize target data

    except ValueError as e:
        print(f"Error: {e}")
        return

    learning_rate = 0.001  # Learning rate for gradient descent
    max_iterations = 100000  # Maximum number of iterations for training
    tolerance = 1e-6  # Convergence threshold: the minimum change in cost function to continue iterations

    print("Training the model...")
    theta0, theta1 = train_model(x, y, learning_rate, max_iterations, tolerance)  # Train the model
    print(f"Training complete. Theta0 = {theta0:.6f}, Theta1 = {theta1:.6f}")

    plot_data_and_model(x, y, theta0, theta1)  # Plot normalized data and regression line
    plot_data_only(original_x, original_y)  # Plot only original data points

    mean_price = np.mean(original_y)  # Mean of the original target data
    std_price = np.std(original_y)  # Standard deviation of the original target data

    # Save the model parameters and normalization information to a JSON file
    with open("theta_values.json", "w") as file:
        json.dump({"theta0": theta0, "theta1": theta1, "mean": np.mean(original_x), "std": np.std(original_x), "mean_price": mean_price, "std_price": std_price}, file)
    print("Theta values and scaling parameters saved for prediction program.")

if __name__ == "__main__":
    main()