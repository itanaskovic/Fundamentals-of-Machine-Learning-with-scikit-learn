import matplotlib.pyplot as plt
import numpy as np

from scipy.optimize import minimize


def negative_log_likelyhood(v):
    l = 0.0
    f1 = 1.0 /np.sqrt(2.0 * np.pi * v[1])
    f2 = 2.0 * v[2]

    for x in X_data:
        l += np.log(f1 * np.exp(-np.square(x - v[0]) / f2))
    return -l


if __name__ == '__main__':
    nb_samples = 100
    X_data =  np.random.normal(loc=0.0, scale=np.sqrt(2.0), size=nb_samples)
    plt.plot(X_data)
    plt.show()
    # WTF?! why next line doesn't work?
    minimize(fun=negative_log_likelyhood, x0=[0.0, 1.0])
