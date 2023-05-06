from sklearn.cluster import KMeans
from tensorflow.keras.datasets import cifar10
from sklearn.preprocessing import OneHotEncoder
from sklearn.cluster import KMeans
import keras
from sklearn.metrics import accuracy_score
from sklearn.metrics.cluster import normalized_mutual_info_score
import numpy as np

def loadData():
    # Load data
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    X_train = X_train.astype('float32')
    #Scaling Image Pixel Values
    X_train /= 255

    print('X_train Shape: ', X_train.shape)
    print('Y_train Shape: ', y_train.shape)
    print('---------------------------------------')
    #Reshaping X train & Test for converting 4D array to 2D
    X_train = X_train.reshape(X_train.shape[0], -1)
    Y_train = y_train.reshape(y_train.shape[0],)

    print('X_train Shape: ', X_train.shape)
    print('Y_train Shape: ', Y_train.shape)

    return X_train, Y_train

def cluster_acc(y_true, y_pred):
    """
    Calculate clustering accuracy. Require scikit-learn installed

    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`

    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    from scipy.optimize import linear_sum_assignment
    ind = np.transpose(linear_sum_assignment(w.max() - w))
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size


def clustering(X_train, Y_train):

    km = KMeans(n_clusters=len(np.unique(Y_train)), n_init=20)
    y_pred = km.fit_predict(X_train)

    nmi = normalized_mutual_info_score(Y_train, y_pred)
    acc = cluster_acc(Y_train, y_pred)
    print("nmi: ", nmi)
    print("acc: ", acc)


def main():

    X_train, Y_train = loadData()
    clustering(X_train, Y_train)

if __name__ == '__main__':
    main()


