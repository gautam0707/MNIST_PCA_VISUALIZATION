import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def read_mnist_training_data():
    # MNIST data:
    # column 0 is labels
    # column 1-785 is data, with values 0 .. 255
    # total size of CSV: (42000, 1, 28, 28)
    train = pd.read_csv('train.csv').values.astype(np.float32)
    train = shuffle(train)

    Xtrain = train[:-1000,1:] / 255
    Ytrain = train[:-1000,0].astype(np.int32)

    Xtest = train[-1000:,1:] / 255
    Ytest = train[-1000:,0].astype(np.int32)
    return Xtrain, Ytrain, Xtest, Ytest


def pca_using_sklearn():
    Xtrain, Ytrain, Xtest, Ytest = read_mnist_training_data()
    fig = plt.figure(figsize=(5.5, 3))
    ax = Axes3D(fig, rect=[0, 0, .7, 1], elev=48, azim=134)
    pca = PCA(n_components=3)
    reduced = pca.fit_transform(Xtrain)
    plt.scatter(reduced[:, 0], reduced[:, 1], s=2, c=Ytrain)
    plt.show()

    plt.plot(pca.explained_variance_ratio_)
    plt.show()

    # cumulative variance
    # choose k = number of dimensions that gives us 95-99% variance
    cumulative = []
    last = 0
    for v in pca.explained_variance_ratio_:
        cumulative.append(last + v)
        last = cumulative[-1]
    plt.plot(cumulative)
    plt.show()

# pca_using_sklearn()
def pca_using_numpy():
    Xtrain, Ytrain, Xtest, Ytest = read_mnist_training_data()
    covX = np.cov(Xtrain.T)
    lambdas, Q = np.linalg.eigh(covX)

    # lambdas are sorted from smallest --> largest
    # some may be slightly negative due to precision
    idx = np.argsort(-lambdas)
    lambdas = lambdas[idx]  # sort in proper order
    lambdas = np.maximum(lambdas, 0)  # get rid of negatives
    Q = Q[:, idx]

    # plot the first 2 columns of Z
    Z = Xtrain.dot(Q)
    plt.scatter(Z[:, 0], Z[:, 1], s=10, c=Ytrain, alpha=0.3)
    plt.legend()
    plt.show()

    # plot variances
    plt.plot(lambdas)
    plt.title("Variance of each component")
    plt.show()

    # cumulative variance
    plt.plot(np.cumsum(lambdas))
    plt.title("Cumulative variance")
    plt.show()

pca_using_numpy()
