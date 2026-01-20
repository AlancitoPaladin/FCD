import numpy as np

from utils import Utils as u


def binarize_image(image, threshold=128):
    """
    Binarizes an image using a threshold.

    Args:
        image: Image matrix
        threshold: Threshold value for binarization (0-255)

    Returns:
        Binarized image (0 and 1)
    """
    if threshold is None:
        threshold = np.mean(image)
    return (image > threshold).astype(np.uint8)


def calculate_central_moments(image, p, q):
    """
    Calculates the central moments of order (p,q) of an image.

    Args:
        image: Image matrix
        p, q: Moment orders

    Returns:
        Central moment of order (p,q)
        :param image:
        :param q:
        :param p:
    """
    # Calculate spatial moments m00, m10, m01
    m00 = np.sum(image)

    if m00 == 0:
        return 0

    # Coordinates of each pixel
    rows, columns = image.shape
    y_indices, x_indices = np.mgrid[:rows, :columns]

    # Centroid
    m10 = np.sum(x_indices * image)
    m01 = np.sum(y_indices * image)
    x_center = m10 / m00
    y_center = m01 / m00

    # Central moment
    central_moment = np.sum(((x_indices - x_center) ** p) * ((y_indices - y_center) ** q) * image)

    return central_moment


def calculate_normalized_moments(image):
    """
    Calculates the normalized moments.

    Args:
        image: Image matrix

    Returns:
        List of 7 normalized moments
    """
    # Calculate central moments
    mu00 = calculate_central_moments(image, 0, 0)
    if mu00 == 0:
        return [0] * 7

    mu20 = calculate_central_moments(image, 2, 0)
    mu02 = calculate_central_moments(image, 0, 2)
    mu11 = calculate_central_moments(image, 1, 1)
    mu30 = calculate_central_moments(image, 3, 0)
    mu03 = calculate_central_moments(image, 0, 3)
    mu21 = calculate_central_moments(image, 2, 1)
    mu12 = calculate_central_moments(image, 1, 2)

    eta20 = mu20 / (mu00 ** (1 + (2 + 0) / 2))
    eta02 = mu02 / (mu00 ** (1 + (0 + 2) / 2))
    eta11 = mu11 / (mu00 ** (1 + (1 + 1) / 2))

    eta30 = mu30 / (mu00 ** (1 + (3 + 0) / 2))
    eta03 = mu03 / (mu00 ** (1 + (0 + 3) / 2))
    eta21 = mu21 / (mu00 ** (1 + (2 + 1) / 2))
    eta12 = mu12 / (mu00 ** (1 + (1 + 2) / 2))

    return [eta20, eta02, eta11, eta30, eta03, eta21, eta12]


def calculate_hu_moments(image, eps=1e-12):
    eta = calculate_normalized_moments(image)

    h1 = eta[0] + eta[1]
    h2 = (eta[0] - eta[1]) ** 2 + 4 * eta[2] ** 2
    h3 = (eta[3] - 3 * eta[4]) ** 2 + (3 * eta[5] - eta[6]) ** 2
    h4 = (eta[3] + eta[4]) ** 2 + (eta[5] + eta[6]) ** 2
    h5 = (eta[3] - 3 * eta[4]) * (eta[3] + eta[4]) * ((eta[3] + eta[4]) ** 2 - 3 * (eta[5] + eta[6]) ** 2) + \
         (3 * eta[5] - eta[6]) * (eta[5] + eta[6]) * (3 * (eta[3] + eta[4]) ** 2 - (eta[5] + eta[6]) ** 2)
    h6 = (eta[0] - eta[1]) * ((eta[3] + eta[4]) ** 2 - (eta[5] + eta[6]) ** 2) + \
         4 * eta[2] * (eta[3] + eta[4]) * (eta[5] + eta[6])
    h7 = (3 * eta[5] - eta[6]) * (eta[3] + eta[4]) * ((eta[3] + eta[4]) ** 2 - 3 * (eta[5] + eta[6]) ** 2) - \
         (eta[3] - 3 * eta[4]) * (eta[5] + eta[6]) * (3 * (eta[3] + eta[4]) ** 2 - (eta[5] + eta[6]) ** 2)

    hu_moments = [h1, h2, h3, h4, h5, h6, h7]
    hu_moments_log = [-np.sign(h) * np.log10(abs(h) + eps) for h in hu_moments]

    return hu_moments_log


def process_images(images, n_samples=None):
    """
    Processes a set of images, binarizes them and calculates their Hu moments.

    Args:
        images: Array with the images
        n_samples: Number of samples to process (None for all)

    Returns:
        Array with the Hu moments of each image
    """
    if n_samples is None:
        n_samples = len(images)
    else:
        n_samples = min(n_samples, len(images))

    all_hu_moments = []

    for i in range(n_samples):
        # Binarize the image
        bin_image = binarize_image(images[i])

        # Calculate Hu moments
        hu_moments = calculate_hu_moments(bin_image)

        all_hu_moments.append(hu_moments)

    return np.array(all_hu_moments)


# Example of use
if __name__ == "__main__":
    # Load the test dataset
    test_path = "../mnist/mnist_test.csv"
    train_path = "../mnist/mnist_train.csv"
    test_images, test_labels = u.load_dataset(test_path)

    # Select an image to visualize
    index = 5  # You can change this value to see different digits
    image = test_images[index]
    label = test_labels[index]

    # Binarize the image
    bin_image = binarize_image(image)

    # Calculate Hu moments for this image
    hu_moments = calculate_hu_moments(bin_image)
    print(f"Hu moments for digit {label}:")
    for i, moment in enumerate(hu_moments, 1):
        print(f"H{i}: {moment:.6f}")

    # Process several images and calculate their Hu moments
    n_samples = 50000  # Number of images to process
    all_moments = process_images(test_images, n_samples)
