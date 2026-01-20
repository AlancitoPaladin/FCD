import matplotlib.pyplot as plt
import numpy as np

from utils import Utils as ut


class LBP:

    @staticmethod
    def calculate(image):
        height, width = image.shape
        lbp = np.zeros((height, width), dtype=np.uint8)

        # 8-neighborhood offsets
        dx = [-1, -1, -1, 0, 0, 1, 1, 1]
        dy = [-1, 0, 1, -1, 1, -1, 0, 1]

        for y in range(1, height - 1):
            for x in range(1, width - 1):
                center = image[y][x]
                pattern = 0

                for i in range(8):
                    neighbor = image[y + dy[i]][x + dx[i]]
                    if neighbor >= center:
                        pattern |= 1 << i
                lbp[y][x] = pattern

        return lbp

    def histogram(image, normalize=True):
        hist, _ = np.histogram(image.ravel(), bins=256, range=(0, 256))
        if normalize:
            hist = hist.astype("float")
            hist /= hist.sum()
        return hist


if __name__ == "__main__":
    image = "../images/example.jpg"
    test_image = ut.load_image(image)

    lbp_image = LBP.calculate(test_image)
    lbp_hist = LBP.histogram(lbp_image, normalize=True)

    print("LBP image:\n", lbp_image)
    print("\nLBP histogram:\n", lbp_hist)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(test_image, cmap="gray")
    axes[0].set_title("Original")

    axes[1].imshow(lbp_image, cmap="gray")
    axes[1].set_title("LBP")

    axes[2].plot(lbp_hist)
    axes[2].set_title("LBP Histogram (normalized)")

    for ax in axes:
        ax.axis("off")

    plt.show()
