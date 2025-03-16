import numpy as np
import logging
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def load_sample_data():
    """
    Generates sample data for training and testing.
    Returns:
        tuple: Training and testing datasets.
    """
    X = np.array([[1], [2], [3], [4], [5]])
    y = np.array([1, 3, 2, 3, 5])
    return train_test_split(X, y, test_size=0.2, random_state=42)

def train_linear_regression(X_train, y_train):
    """
    Trains a Linear Regression model.
    Returns:
        model: Trained Linear Regression model.
    """
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """
    Evaluates the trained model.
    """
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    logging.info(f"Mean Squared Error: {mse:.4f}")
    logging.info(f"Predictions: {y_pred}")

def main():
    """
    Main function to load data, train, and evaluate the Linear Regression model.
    """
    X_train, X_test, y_train, y_test = load_sample_data()
    model = train_linear_regression(X_train, y_train)
    evaluate_model(model, X_test, y_test)

if __name__ == "__main__":
    main()
