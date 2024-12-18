import json
import numpy as np
from linear_regression.tools import normalize_prediction_input, denormalize_prediction_output 


# --------------------------------------------------
def predict_price(mileage, theta0, theta1):
    """
    Predict the price of a car based on mileage.
    :param mileage: Mileage of the car (in km)
    :param theta0: Intercept term from trained model
    :param theta1: Slope term from trained model
    :return: Predicted price
    """
    return theta0 + (theta1 * mileage)

# --------------------------------------------------
def main():
    """
    Main function to load model parameters and predict car price.
    """
    try:
        # Load theta values and scaling parameters from the saved file
        with open("theta_values.json", "r") as file:
            data = json.load(file)
        theta0 = data["theta0"]
        theta1 = data["theta1"]
        mean_mileage = data["mean"]
        std_mileage = data["std"]
        mean_price = data.get("mean_price", 0)  # Default to 0 if not found
        std_price = data.get("std_price", 1)   # Default to 1 if not found
    except FileNotFoundError:
        print("Error: Model parameters not found. Please run the training program first.")
        return
    except json.JSONDecodeError:
        print("Error: Failed to decode model parameters. Check the file format.")
        return

    # Get user input
    try:
        mileage = float(input("Enter the mileage of the car (in km): "))
    except ValueError:
        print("Invalid input. Please enter a numeric value.")
        return

    # Normalize the input mileage
    normalized_mileage = normalize_prediction_input(mileage, mean_mileage, std_mileage)

    # Predict the normalized price
    normalized_price = predict_price(normalized_mileage, theta0, theta1)

    # Denormalize the price
    price = denormalize_prediction_output(normalized_price, mean_price, std_price)
    print(f"Estimated price for a car with mileage {mileage:.2f} km is: {price:.2f}")

if __name__ == "__main__":
    main()
