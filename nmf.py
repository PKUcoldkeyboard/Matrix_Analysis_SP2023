import os
import numpy as np
from sklearn.decomposition import NMF
from PIL import Image
from combine import load_images, combine_images


def main():
    images = load_images("./datasets/images")
    M = combine_images(images)
    # experience parameter
    d = 16
    width, height = 64, 64

    model = NMF(n_components=d, init='random',
                random_state=0, tol=1e-6, max_iter=200)
    W = model.fit_transform(M)
    H = model.components_

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
