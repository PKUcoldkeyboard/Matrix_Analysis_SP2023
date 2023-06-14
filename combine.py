import os
import numpy as np
from PIL import Image

def load_images(image_folder, size=(64, 64)):
    """
    Load and resize images from a specified folder.

    Parameters
    ----------
    image_folder : str
        Path to the directory containing images.
    size : tuple of int, optional
        The size to which all images will be resized (default is (64, 64)).

    Returns
    -------
    images : list of PIL.Image
        List of resized images.
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
        List of images.
    K : int, optional
        The number of pixels in the resulting image (default is 4096).

    Returns
    -------
    M : np.ndarray
        Matrix with shape (K, 3N), where K equals Width x Height, and N is the number of images.
    """
    images_matrix = []
    for img in images:
        img_array = np.array(img)
        assert img_array.shape[2] == 3, "Image does not have 3 channels."
        r, g, b = img_array[:, :, 0], img_array[:, :, 1], img_array[:, :, 2]
        images_matrix.extend([r.flatten(), g.flatten(), b.flatten()])
    M = np.array(images_matrix).T
    return M
