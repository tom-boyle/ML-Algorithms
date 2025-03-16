import numpy as np
import logging
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def load_sample_data():
    """
    Generates sample data for PCA analysis.
    Returns:
        np.ndarray: Sample data points.
    """
    return np.array([[2.5, 2.4], [0.5, 0.7], [2.2, 2.9],
                     [1.9, 2.2], [3.1, 3.0], [2.3, 2.7],
                     [2, 1.6], [1, 1.1], [1.5, 1.6], [1.1, 0.9]])

def apply_pca(X, n_components=2):
    """
    Applies Principal Component Analysis (PCA) to reduce dimensions.
    Returns:
        np.ndarray: Transformed principal components.
    """
    pca = PCA(n_components=n_components)
    return pca.fit_transform(X)

def plot_pca(principal_components):
    """
    Plots the principal components.
    """
    plt.scatter(principal_components[:, 0], principal_components[:, 1])
    plt.title('PCA Analysis')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.show()

def main():
    """
    Main function to load data, apply PCA, and plot the results.
    """
    X = load_sample_data()
    principal_components = apply_pca(X)
    logging.info("PCA transformation completed.")
    plot_pca(principal_components)

if __name__ == "__main__":
    main()
