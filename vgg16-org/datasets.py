import numpy as np


def extract_vgg16_features(x):
    from keras.utils import img_to_array, array_to_img
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

    print(features)

    return features

def load_mnist():
    # the data, shuffled and split between train and test sets
    from tensorflow.keras.datasets import mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x = np.concatenate((x_train, x_test))
    y = np.concatenate((y_train, y_test))
    x = x.reshape([-1, 28, 28, 1]) / 255.0
    print('MNIST samples', x.shape)
    return x, y


def load_mnist_test():
    # the data, shuffled and split between train and test sets
    from tensorflow.keras.datasets import mnist
    _, (x, y) = mnist.load_data()
    x = x.reshape([-1, 28, 28, 1]) / 255.0
    print('MNIST samples', x.shape)
    return x, y


def load_fashion_mnist():
    from tensorflow.keras.datasets import fashion_mnist  # this requires keras>=2.0.9
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    x = np.concatenate((x_train, x_test))
    y = np.concatenate((y_train, y_test))
    x = x.reshape([-1, 28, 28, 1]) / 255.0
    print('Fashion MNIST samples', x.shape)
    return x, y


def load_usps(data_path='./data/usps'):
    import os
    if not os.path.exists(data_path+'/usps_train.jf'):
        raise ValueError("No data for usps found, please download the data from links in \"./data/usps/download_usps.txt\".")

    with open(data_path + '/usps_train.jf') as f:
        data = f.readlines()
    data = data[1:-1]
    data = [list(map(float, line.split())) for line in data]
    data = np.array(data)
    data_train, labels_train = data[:, 1:], data[:, 0]

    with open(data_path + '/usps_test.jf') as f:
        data = f.readlines()
    data = data[1:-1]
    data = [list(map(float, line.split())) for line in data]
    data = np.array(data)
    data_test, labels_test = data[:, 1:], data[:, 0]

    x = np.concatenate((data_train, data_test)).astype('float64') / 2.
    y = np.concatenate((labels_train, labels_test))
    x = x.reshape([-1, 16, 16, 1])
    print('USPS samples', x.shape)
    return x, y


def load_stl():
    from tensorflow import keras
    import tensorflow as tf
    import numpy as np
    from keras.preprocessing.image import ImageDataGenerator
    from keras.layers import Rescaling

    train_ds = keras.utils.image_dataset_from_directory(
        directory='./data/stl/images/train/',
        labels='inferred',
        batch_size=16,
        image_size=(96, 96),
        shuffle=True,
        seed=None,
        validation_split=None,
    )

    images = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
    labels = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True)

    for x, y in train_ds.unbatch():
        images = images.write(images.size(), np.divide(x, 255.0))
        labels = labels.write(labels.size(), y)

    x = tf.stack(images.stack(), axis=0)
    labels = tf.stack(labels.stack(), axis=0)

    y = [tf.keras.backend.get_value(l) for l in labels]

    return x.numpy(), np.asarray(y, dtype=np.int8)


def load_data_conv(dataset):
    if dataset == 'mnist':
        return load_mnist()
    elif dataset == 'mnist-test':
        return load_mnist_test()
    elif dataset == 'fmnist':
        return load_fashion_mnist()
    elif dataset == 'usps':
        return load_usps()
    elif dataset == 'stl':
        return load_stl()
    else:
        raise ValueError('Not defined for loading %s' % dataset)


def load_data(dataset):
    x, y = load_data_conv(dataset)
    return x.reshape([x.shape[0], -1]), y

