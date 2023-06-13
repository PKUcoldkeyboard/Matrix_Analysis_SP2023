import os
import numpy as np
from sklearn.decomposition import NMF
from PIL import Image
from combine import load_images, combine_images


class MyNMF():
    """ Non-negative Matrix Factorization (NMF)
    
    Find two non-negative matrices (W, H) whose product approximates the non-negative matrix X.
    
    Parameters
    ----------
    r : int, default=16
        Rank of the low-rank matrix.
        
    max_iter : int, default=200
        Maximum number of iterations before timing out.
        
    tol : float, default=1e-4
        Tolerance of the stopping condition.
        
    random_state : int, RandomState instance, default=None
        Determines random number generation for initialization. Use an int to make the randomness deterministic.
        
    """
    def __init__(self, r=16, max_iter=200, tol=1e-4, random_state=None):
        self.r = r
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

    def fit_transform(self, X):
        # initialize W and H
        np.random.seed(self.random_state)
        W = np.random.rand(X.shape[0], self.r)
        H = np.random.rand(self.r, X.shape[1])
        W[W < 0] = 0
        W[W > 255] = 255
        H[H < 0] = 0
        H[H > 255] = 255

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
    images = load_images("./datasets/images")
    M = combine_images(images)
    # experience parameter
    d = 16
    width, height = 64, 64

    model = MyNMF(r=d, max_iter=200, tol=1e-4, random_state=0)
    W, H = model.fit_transform(M)

    # show images of column vectors of low-rank matrix U
    for i in range(d):
        img = W[:, i].reshape(height, width)
        img = (img - img.min()) / (img.max() - img.min()) * 255
        img = Image.fromarray(img.astype(np.uint8), mode='L')
        img.save(os.path.join("./results/nmf", f"W_{i}.jpg"))

    # reconstructed first 20 images
    M_d = W @ H

    N = int(M_d.shape[1] / 3)
    for i in range(20):
        r = M_d[:, i * 3].reshape(height, width)
        g = M_d[:, i * 3 + 1].reshape(height, width)
        b = M_d[:, i * 3 + 2].reshape(height, width)
        # concatenate the three channels into a single image
        img = np.dstack((r, g, b)).reshape(height, width, 3)
        img[img > 255] = 255
        img[img < 0] = 0
        img = Image.fromarray(img.astype(np.uint8), mode='RGB')
        img.save(os.path.join("./results/nmf", f"reconst_{i}.jpg"))


if __name__ == "__main__":
    main()
