import numpy as np
import tensorflow as tf
import logging
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def load_data():
    """
    Loads and preprocesses the MNIST dataset.
    Returns:
        tuple: Training and test datasets.
    """
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape((x_train.shape[0], 28 * 28)).astype("float32") / 255
    x_test = x_test.reshape((x_test.shape[0], 28 * 28)).astype("float32") / 255
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    return (x_train, y_train), (x_test, y_test)

def build_model():
    """
    Builds and compiles a simple neural network model.
    Returns:
        model (Sequential): Compiled model.
    """
    model = Sequential([
        Dense(512, activation='relu', input_shape=(28 * 28,)),
        Dense(256, activation='relu'),
        Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def train_and_evaluate(model, x_train, y_train, x_test, y_test):
    """
    Trains and evaluates the model.
    """
    model.fit(x_train, y_train, epochs=10, batch_size=128, validation_split=0.2)
    test_loss, test_acc = model.evaluate(x_test, y_test)
    logging.info(f"Test accuracy: {test_acc:.4f}")

def main():
    """
    Main function to load data, build, train, and evaluate the model.
    """
    (x_train, y_train), (x_test, y_test) = load_data()
    model = build_model()
    train_and_evaluate(model, x_train, y_train, x_test, y_test)

if __name__ == "__main__":
    main()
