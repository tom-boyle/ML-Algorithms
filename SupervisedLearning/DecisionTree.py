import numpy as np
import logging
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def load_sample_data():
    """
    Generates sample data for training and testing.
    Returns:
        tuple: Training and testing datasets.
    """
    X = np.array([[1], [2], [3], [4], [5], [6]])
    y = np.array([0, 1, 0, 1, 0, 1])
    return train_test_split(X, y, test_size=0.2, random_state=42)

def train_decision_tree(X_train, y_train):
    """
    Trains a Decision Tree model.
    Returns:
        model: Trained Decision Tree classifier.
    """
    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """
    Evaluates the trained model.
    """
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    logging.info(f"Model Accuracy: {accuracy:.4f}")
    logging.info(f"Predictions: {y_pred}")

def main():
    """
    Main function to load data, train, and evaluate the Decision Tree model.
    """
    X_train, X_test, y_train, y_test = load_sample_data()
    model = train_decision_tree(X_train, y_train)
    evaluate_model(model, X_test, y_test)

if __name__ == "__main__":
    main()
