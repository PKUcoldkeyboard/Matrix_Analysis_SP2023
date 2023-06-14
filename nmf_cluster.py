import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

class ClusterNMF():
    """ Non-negative Matrix Factorization (NMF)

    Find two non-negative matrices (W, H) whose product approximates the non-negative matrix X.

    Parameters
    ----------
    num_cluster : int, default=None
        The number of clusters to consider.

    max_iter : int, default=200
        Maximum number of iterations before timing out.

    tol : float, default=1e-4
        Tolerance of the stopping condition.

    random_state : int, RandomState instance, default=None
        Determines random number generation for initialization. Use an int to make the randomness deterministic.

    """

    def __init__(self, num_cluster=None, max_iter=200, tol=1e-4, random_state=None):
        self.num_cluster = num_cluster
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

    def fit_transform(self, X):
        # initialize W and H
        np.random.seed(self.random_state)
        X = X.to_numpy()
        W = np.random.rand(X.shape[0], self.num_cluster)
        H = np.random.rand(self.num_cluster, X.shape[1])

        prev_loss = np.inf
        for i in range(self.max_iter + 1):
            # multiplicative update rules
            H = H * (W.T @ X) / (W.T @ W @ H + 1e-12)
            W = W * (X @ H.T) / (W @ H @ H.T + 1e-12)

            loss = np.linalg.norm(X - W @ H, ord='fro')
            if i % 10 == 0:
                print(f'Epoch: {i}, loss: {loss}')

            # Check for convergence
            delta_loss = np.abs(prev_loss - loss)
            if delta_loss < self.tol:
                break

            prev_loss = loss

        return W, H


def main():
    df = pd.read_csv('datasets/ebs/waveform-5000.csv')

    X, y = df.iloc[:, :-1], df.iloc[:, -1]
    print(X.shape, y.shape)
    # Normalize the data
    X = (X - X.min()) / (X.max() - X.min())

    model = ClusterNMF(num_cluster=3, random_state=42)
    W, _ = model.fit_transform(X)

    # Normalize the weights
    W = W / W.sum(axis=1, keepdims=True)
    # Choose the cluster with the highest weight
    clusters = np.argmax(W, axis=1)

    # PCA for visualization
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    
    # Create a dataframe for visualization
    df_plot = pd.DataFrame(X_pca, columns=['PC 1', 'PC 2'])
    df_plot['Cluster'] = pd.Series(clusters, index=df_plot.index)

    # Plot the clusters
    sns.scatterplot(data=df_plot, x='PC 1', y='PC 2', hue='Cluster', palette='viridis')
    plt.title('Waveform Clustering visualization')
    plt.savefig('results/ebs/nmf_cluster.png')
    plt.show()


if __name__ == '__main__':
    main()
