import os
import numpy as np
from PIL import Image


def load_images(image_folder, size=(64, 64)):
    """
    Load images and reshape them from a given folder.

    Parameters
    ----------
    image_folder : str
        The folder where the images are stored.

    Returns
    -------
    list of PIL.Image
        The list of images.
    """
    images = []

    for filename in os.listdir(image_folder):
        img = Image.open(os.path.join(image_folder, filename))
        img = img.resize(size)
        images.append(img)

    return images


def combine_images(images, K=4096):
    """
    Combine a list of images into a matrix.

    Parameters
    ----------
    images : list of PIL.Image
        The list of images.

    Returns
    -------
    np.ndarray
        Matrix M with shape (K, 3N) - K = Width x Height, N = number of images.
    """
    images_matrix = []

    for img in images:
        img_array = np.array(img)

        # Ensure the image has 3 channels
        assert img_array.shape[2] == 3, "Image does not have 3 channels."

        # Split the image into R, G, B channels
        r, g, b = img_array[:, :, 0], img_array[:, :, 1], img_array[:, :, 2]

        # Flatten the channels and append them to the images_matrix
        images_matrix.extend([r.flatten(), g.flatten(), b.flatten()])

    M = np.array(images_matrix).T

    return M
