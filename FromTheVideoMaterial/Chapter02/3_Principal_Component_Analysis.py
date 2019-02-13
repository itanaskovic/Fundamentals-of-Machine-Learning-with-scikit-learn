from sklearn.datasets import load_digits
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    digits = load_digits()

    selection = np.random.randint(0, 100, size=100)

    fig, ax = plt.subplots(10, 10, figsize=(7, 7))

    samples = [digits.data[x].reshape((8, 8)) for x in selection]

    for i in range(10):
        for j in range(10):
            ax[i, j].set_axis_off()
            ax[i, j].imshow(samples[(i * 8) + j], cmap='gray')

    plt.show(block=True)

    pca = PCA(n_components=36, whiten=True)
    X_pca = pca.fit_transform(digits.data / 255)

    print(pca.explained_variance_ratio_)

    # let's plot this ratio
    fig, ax = plt.subplots(1, 2, figsize=(16, 6))

    ax[0].set_xlabel('Component')
    ax[0].set_ylabel('Variance ratio (%)')
    ax[0].bar(np.arange(36), pca.explained_variance_ratio_ * 100.0)

    ax[1].set_xlabel('Component')
    ax[1].set_ylabel('Cumulative variance (%)')
    ax[1].bar(np.arange(36), np.cumsum(pca.explained_variance_)[::-1])

    plt.show()



