import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def normalize_data(data):
    # Normalize data to have mean 0 and standard deviation 1
    return (data - np.mean(data)) / np.std(data)

def plot_data_only(x, y):
    # Plot only the original (non-normalized) data points
    plt.scatter(x, y, color='blue', label='Original Data Points')
    plt.title('Original Data Points')
    plt.xlabel('X (Input Feature)')
    plt.ylabel('Y (Target)')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_data_and_model(x, y, theta0, theta1):
    # Plot the normalized data points along with the regression line
    plt.scatter(x, y, color='blue', label='Data Points')
    regression_line = theta0 + theta1 * x  # Regression line based on theta values
    plt.plot(x, regression_line, color='red', label='Regression Line')
    plt.title('Data with Linear Regression')
    plt.xlabel('Normalized X (Input Feature)')
    plt.ylabel('Normalized Y (Target)')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_normalized_regression(x, y, theta0, theta1, title, iteration=None, cost=None):
    # Unified function to plot data and regression line
    plt.scatter(x, y, color='blue', label='Data Points')
    regression_line = theta0 + theta1 * x  # Regression line based on theta values
    plt.plot(x, regression_line, color='red', label='Regression Line')
    plt.title(title if iteration is None else f'{title} (Iteration {iteration}, Cost = {cost:.6f})')
    plt.xlabel('Normalized X (Input Feature)')
    plt.ylabel('Normalized Y (Target)')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_cost_function(cost_history):
    # Plot cost function decrease over iterations
    plt.plot(range(len(cost_history)), cost_history, color='green')
    plt.title('Cost Function Decrease Over Iterations')
    plt.xlabel('Iterations')
    plt.ylabel('Cost (MSE)')
    plt.grid(True)
    plt.show()

def plot_data_with_regression(x, y, theta0, theta1, x_mean, x_std, y_mean, y_std):
    # Plot original (non-normalized) data points with properly scaled regression line
    plt.scatter(x, y, color='blue', label='Original Data Points')
    regression_line = (theta0 * y_std + y_mean) + (theta1 * y_std / x_std) * (x - x_mean)  # Scale back theta values to original data
    plt.plot(x, regression_line, color='red', label='Regression Line')
    plt.title('Original Data with Linear Regression Line')
    plt.xlabel('X (Input Feature)')
    plt.ylabel('Y (Target)')
    plt.legend()
    plt.grid(True)
    plt.show()

def read_csv(file_path):
    """
    Read mileage (km) and price data from a CSV file using Pandas with validation.

    Validations:
    - File extension check
    - File existence and format validation
    - Column presence validation
    - Missing value handling
    - Numeric value validation
    - Range checks for unrealistic values
    """
    # Step 1: Ensure the file has a .csv extension
    # This prevents non-CSV files from being processed.
    if not str(file_path).endswith('.csv'):
        raise ValueError("The file must have a .csv extension.")

    try:
        # Step 2: Attempt to read the CSV file into a DataFrame
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        # Raise an error if the file is not found
        raise FileNotFoundError(f"The file at '{file_path}' was not found.")
    except pd.errors.EmptyDataError:
        # Raise an error if the file is empty
        raise ValueError("The file is empty. Please provide a valid dataset.")
    except pd.errors.ParserError:
        # Raise an error if the file cannot be parsed as a valid CSV
        raise ValueError("The file could not be parsed. Ensure it is in a valid CSV format.")

    # Step 3: Validate that required columns ('km' and 'price') are present
    required_columns = {"km", "price"}
    if not required_columns.issubset(df.columns):
        raise ValueError(f"The CSV file must contain the following columns: {required_columns}")

    # Step 4: Check for and handle missing values in 'km' and 'price'
    if df[["km", "price"]].isnull().values.any():
        # Drop rows with missing values and raise a warning
        df = df.dropna(subset=["km", "price"])
        raise ValueError("Warning: Missing values detected. These rows will be dropped.")

    # Step 5: Validate that 'km' and 'price' columns contain numeric values
    if not pd.api.types.is_numeric_dtype(df["km"]) or not pd.api.types.is_numeric_dtype(df["price"]):
        raise ValueError("Both 'km' and 'price' columns must contain numeric values.")

    # Step 6: Validate that 'km' and 'price' have non-negative values
    if (df["km"] < 0).any() or (df["price"] < 0).any():
        raise ValueError("Both 'km' and 'price' must have non-negative values.")

    # Step 7: Check for unrealistic values in 'price' and 'km'
    if (df["price"] > 1e7).any():
        raise ValueError("Detected unrealistic values in the 'price' column. Please check your data.")
    if (df["km"] > 1e6).any():
        raise ValueError("Detected unrealistic values in the 'km' column. Please check your data.")

    # Step 8: Convert valid 'km' and 'price' data to lists for further processing
    km = df["km"].to_list()
    price = df["price"].to_list()

    # Success message with the number of rows loaded
    print(f"Successfully loaded {len(km)} rows of data.")
    return km, price
