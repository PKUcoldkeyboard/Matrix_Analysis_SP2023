import os
import numpy as np
from PIL import Image
from combine import load_images, combine_images


def main():
    images = load_images("./datasets/images")
    M = combine_images(images)

    d = 16
    width, height = 64, 64

    U, S, V = np.linalg.svd(M, full_matrices=False)

    # show images of column vectors of low-rank matrix U
    for i in range(d):
        img = U[:, i].reshape(height, width)
        img = (img - img.min()) / (img.max() - img.min()) * 255
        img = Image.fromarray(img.astype(np.uint8), mode='L')
        img.save(os.path.join("./results/svd", f"U_{i}.jpg"))

    # reconstructed first 20 images
    U_d = U[:, :d]
    S_d = S[:d]
    V_d = V[:d, :]
    M_d = U_d @ np.diag(S_d) @ V_d
    
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
        img.save(os.path.join("./results/svd", f"reconst_{i}.jpg"))


if __name__ == "__main__":
    main()
