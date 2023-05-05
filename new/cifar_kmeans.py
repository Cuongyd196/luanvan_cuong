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

def clustering(X_train, Y_train):

    km = KMeans(n_clusters=len(np.unique(Y_train)), n_init=20)
    y_pred = km.fit_predict(X_train)
    print('y_pred', y_pred)
    print(y_pred.shape, 'y_predshape')
    print(y_pred.size, 'y_size')

    print('Y_train', Y_train)
    print(Y_train.shape, 'Y_train shape')
    print(Y_train.size, 'Y_train_size')

    accuracy = normalized_mutual_info_score(Y_train, y_pred)
    print("Accuracy: ", accuracy)


def main():

    X_train, Y_train = loadData()
    clustering(X_train, Y_train)

if __name__ == '__main__':
    main()


