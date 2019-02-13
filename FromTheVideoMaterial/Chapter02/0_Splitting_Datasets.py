from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.utils import check_random_state


if __name__ == '__main__':
    boston = load_boston()
    X = boston.data
    Y = boston.target
    print(X.shape)
    print(Y.shape)

    X_train, Y_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=1000)

    # better solution
    rs = check_random_state(1000)
    X_train, Y_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=rs)

