import os
import numpy as np
from PIL import Image
from combine import load_images, combine_images

def main():
    # Load images from the specified directory and combine them into a matrix
    images = load_images("./datasets/images")
    M = combine_images(images)
    
    # Specify the number of principal components to retain
    d = 16
    width, height = 64, 64

    # Perform singular value decomposition (SVD) on the matrix M
    U, S, V = np.linalg.svd(M, full_matrices=False)
    
    print("Singular values:", S[:d])

    # Display images corresponding to the column vectors of low-rank matrix U
    for i in range(d):
        img = U[:, i].reshape(height, width)
        img = (img - img.min()) / (img.max() - img.min()) * 255
        img = Image.fromarray(img.astype(np.uint8), mode='L')
        img.save(os.path.join("./results/svd", f"U_{i}.jpg"))

    # Reconstruct first 20 images using the first d singular values/vectors
    U_d = U[:, :d]
    S_d = S[:d]
    V_d = V[:d, :]
    M_d = U_d @ np.diag(S_d) @ V_d

    # Split each combined image in the matrix M_d into three channels (R, G, B)
    N = int(M_d.shape[1] / 3)
    for i in range(20):
        r = M_d[:, i * 3].reshape(height, width)
        g = M_d[:, i * 3 + 1].reshape(height, width)
        b = M_d[:, i * 3 + 2].reshape(height, width)
        
        # Concatenate the three channels into a single image
        img = np.dstack((r, g, b)).reshape(height, width, 3)
        img[img > 255] = 255
        img[img < 0] = 0
        img = Image.fromarray(img.astype(np.uint8), mode='RGB')
        img.save(os.path.join("./results/svd", f"reconst_{i}.jpg"))

if __name__ == "__main__":
    main()
