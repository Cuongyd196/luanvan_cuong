import numpy as np
from keras.datasets import cifar10
from keras.applications.vgg16 import VGG16
from keras.layers import Flatten
from sklearn.cluster import KMeans
from sklearn import metrics
import tensorflow as tf
import keras
physical_devices = tf.config.list_physical_devices('GPU') 
tf.config.experimental.set_memory_growth(physical_devices[0], True)

def extract_vgg16_features(x):
    from keras.preprocessing.image import img_to_array, array_to_img
    from keras.applications.vgg16 import preprocess_input, VGG16
    from keras.models import Model
    im_h = 224
    model = VGG16(include_top=True, weights='imagenet', input_shape=(im_h, im_h, 3))
    feature_model = Model(model.input, model.get_layer('fc1').output)
    print('extracting features...')
    x = np.asarray([img_to_array(array_to_img(im, scale=False).resize((im_h,im_h))) for im in x])
    x = preprocess_input(x)  # data - 127. #data/255.#
    features = feature_model.predict(x)
    print('Features shape = ', features.shape)

    return features



def load_cifar10(data_path='./data/cifar10'):
    (train_x, train_y), (test_x, test_y) = cifar10.load_data()
    x = np.concatenate((train_x, test_x))
    y = np.concatenate((train_y, test_y)).reshape((60000,))

    x= x[:20000]
    y= y[:20000]

    # if features are ready, return them
    # import os.path
    # if os.path.exists(data_path + '/cifar10_features.npy'):
    #     return np.load(data_path + '/cifar10_features.npy'), y

    # extract features
    features = np.zeros((20000, 4096))
    for i in range(1):
        idx = range(i*10000, (i+1)*10000)
        print("The %dth 10000 samples" % i)
        features[idx] = extract_vgg16_features(x[idx])

    # scale to [0,1]
    from sklearn.preprocessing import MinMaxScaler
    features = MinMaxScaler().fit_transform(features)

    # save features
    #np.save(data_path + '/cifar10_features.npy', features)
    #print('features saved to ' + data_path + '/cifar10_features.npy')

    return features, y

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
    from scipy.optimize import linear_sum_assignment as linear_assignment
    ind = linear_assignment(w.max() - w)
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size



def main():
    x, y = load_cifar10()
    # clustering
    km = KMeans(n_clusters=len(np.unique(y)), n_init=20)
    y_pred = km.fit_predict(x)
    # acc = cluster_acc(y, y_pred)
    # evaluate
    # acc = metrics.accuracy_score(y, y_pred)
    nmi = metrics.normalized_mutual_info_score(y, y_pred)
    ari = metrics.adjusted_rand_score(y, y_pred)
    print('nmi:', nmi)
    print('ari:', ari)


if __name__ == '__main__':
    main()

