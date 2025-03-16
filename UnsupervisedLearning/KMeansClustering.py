import numpy as np
import logging
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def load_sample_data():
    """
    Generates sample data for clustering.
    Returns:
        np.ndarray: Sample data points.
    """
    return np.array([[1, 2], [1, 4], [1, 0], [4, 2], [4, 4], [4, 0]])

def train_kmeans(X, n_clusters=2):
    """
    Trains a K-Means clustering model.
    Returns:
        model: Trained K-Means model.
        labels: Cluster labels for data points.
    """
    model = KMeans(n_clusters=n_clusters, random_state=0)
    labels = model.fit_predict(X)
    return model, labels

def plot_clusters(X, model, labels):
    """
    Plots the clustered data and centroids.
    """
    plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
    plt.scatter(model.cluster_centers_[:, 0], model.cluster_centers_[:, 1], s=200, c='red', marker='X')
    plt.title('K-Means Clustering')
    plt.show()

def main():
    """
    Main function to load data, train the model, and plot the clusters.
    """
    X = load_sample_data()
    model, labels = train_kmeans(X)
    logging.info("K-Means clustering completed.")
    plot_clusters(X, model, labels)

if __name__ == "__main__":
    main()
