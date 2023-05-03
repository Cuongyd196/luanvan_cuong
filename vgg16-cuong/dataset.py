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

def load_data_conv(dataset):
    if dataset == 'mnist':
        return load_mnist()
    else:
        raise ValueError('Not defined for loading %s' % dataset)

def load_data(dataset):
    x, y = load_data_conv(dataset)
    return x.reshape([x.shape[0], -1]), y

