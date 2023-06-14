import os
import numpy as np
from sklearn.decomposition import NMF
from PIL import Image
from combine import load_images, combine_images


class MyNMF():
    """ 
    Non-negative Matrix Factorization (NMF)
    
    Find two non-negative matrices (W, H) whose product approximates the non-negative matrix X.
    This class implements a custom NMF model.
    """
    def __init__(self, r=16, max_iter=200, tol=1e-4, random_state=None):
        # Initialize parameters for NMF
        self.r = r
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

    def fit_transform(self, X):
        # Initialize W and H with random non-negative values
        np.random.seed(self.random_state)
        W = np.random.rand(X.shape[0], self.r)
        H = np.random.rand(self.r, X.shape[1])

        prev_loss = np.inf
        for i in range(self.max_iter + 1):
            # Apply multiplicative update rules
            H = H * (W.T @ X) / (W.T @ W @ H + 1e-12)
            W = W * (X @ H.T) / (W @ H @ H.T + 1e-12)
                
            # Compute and print the loss every 10 iterations
            loss = np.linalg.norm(X - W @ H, ord='fro')
            if i % 10 == 0:
                print(f'Epoch: {i}, loss: {loss}')
                
            # Check for convergence based on the defined tolerance
            if np.abs(prev_loss - loss) < self.tol:
                break
            
            prev_loss = loss
            
        return W, H
    

def main():
    # Load and combine images
    images = load_images("./datasets/images")
    M = combine_images(images)

    # Define model and fit-transform the data
    model = MyNMF(r=16, max_iter=200, tol=1e-4, random_state=0)
    W, H = model.fit_transform(M)

    # Show images of column vectors of low-rank matrix W
    for i in range(16):
        img = W[:, i].reshape(64, 64)
        img = (img - img.min()) / (img.max() - img.min()) * 255
        img = Image.fromarray(img.astype(np.uint8), mode='L')
        img.save(os.path.join("./results/nmf", f"W_{i}.jpg"))

    # Reconstruct the first 20 images
    M_d = W @ H

    # For each image, reshape the flattened RGB channels and save the reconstructed image
    for i in range(20):
        r = M_d[:, i * 3].reshape(64, 64)
        g = M_d[:, i * 3 + 1].reshape(64, 64)
        b = M_d[:, i * 3 + 2].reshape(64, 64)
        
        img = np.dstack((r, g, b)).reshape(64, 64, 3)
        img[img > 255] = 255
        img[img < 0] = 0
        img = Image.fromarray(img.astype(np.uint8), mode='RGB')
        img.save(os.path.join("./results/nmf", f"reconst_{i}.jpg"))


if __name__ == "__main__":
    main()
