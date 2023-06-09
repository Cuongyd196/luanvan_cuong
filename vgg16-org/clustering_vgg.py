
from time import time

from pkg_resources import yield_lines
import numpy as np
import keras.backend as K
from keras import layers
from keras.layers import Layer, InputSpec
from keras.layers import Dense, Input
from keras.models import Model
from keras.optimizers import SGD
from keras import callbacks
from keras.initializers import VarianceScaling
from sklearn.cluster import KMeans
import metrics

def autoencoder0(im_size, dims, act='relu', init='glorot_uniform'):
    n_stacks = len(dims) - 1
    print('+ n_stacks:', n_stacks)
     # input
    img_input = Input(shape=(im_size, im_size, 3), name='input')
    x = h = img_input
    kernel = (3, 3)

    # internal layers in encoder
    for i in range(n_stacks):        
        h = layers.Conv2D(dims[i], kernel, strides=1, activation=act, padding='same', name='encoder_%d' % i)(h)        
        h = layers.BatchNormalization(momentum=0.8)(h)
        h = layers.MaxPooling2D((2, 2), padding='same')(h)    
    
    # hidden layer, features are extracted from here
    i = n_stacks    
    hidden_shape = (h.shape[1],h.shape[2], dims[i]) 
    hidden_dims = h.shape[1]*h.shape[2]*dims[i]
    print(hidden_shape)    
    h = layers.Flatten()(h)    
    h = Dense(hidden_dims, kernel_initializer=init, name='hidden_layer')(h) 
    y = layers.Reshape(hidden_shape)(h)

    #h = layers.Conv2D(dims[i], kernel, strides=1, activation=act, padding='same', name='hidden_layer')(h)
    #y = h        
    
    # internal layers in decoder
    for i in range(n_stacks-1, -1, -1):        
        y = layers.UpSampling2D((2, 2))(y)
        # y = layers.Conv2D(dims[i], kernel, activation=act, padding='same', name='decoder_%d' % i)(y)        
        y = layers.Conv2DTranspose(dims[i], kernel, strides=1, activation=act, padding='same', name='decoder_%d' % i)(y)
        y = layers.BatchNormalization(momentum=0.8)(y)

    y = layers.Conv2DTranspose(3, kernel, strides=1, activation=act, padding='same', name='decoder_output')(y)

    return Model(inputs=x, outputs=y, name='AE'), Model(inputs=x, outputs=h, name='encoder')


def autoencoder(im_size, dims, act='relu', init='glorot_uniform'):
    n_stacks = len(dims) - 1
    print('+ n_stacks:', n_stacks)
     # input
    img_input = Input(shape=(im_size, im_size, 3), name='input')
    x = h = img_input
    kernel = (3, 3)

    # internal layers in encoder
    for i in range(n_stacks):        
        h = layers.Conv2D(dims[i], kernel, strides=2, activation=act, padding='same', name='encoder_%d' % i)(h)        
        h = layers.BatchNormalization(momentum=0.8)(h)
        #h = layers.MaxPooling2D((2, 2), padding='same')(h)    
    
    # hidden layer, features are extracted from here => don't use any activation    
    i = n_stacks        
    h = layers.Conv2D(dims[i], kernel, strides=1, padding='same', name='hidden_layer')(h)
    y = h            
    # internal layers in decoder
    for i in range(n_stacks-1, -1, -1):        
        #y = layers.UpSampling2D((2, 2))(y)
        # y = layers.Conv2D(dims[i], kernel, activation=act, padding='same', name='decoder_%d' % i)(y)        
        y = layers.Conv2DTranspose(dims[i], kernel, strides=2, activation=act, padding='same', name='decoder_%d' % i)(y)
        y = layers.BatchNormalization(momentum=0.8)(y)

    #output decoder should use sigmoid activaion, use normal conv2d
    y = layers.Conv2D(3, kernel, strides=1, padding='same', activation='sigmoid', name='decoder_output')(y)


    return Model(inputs=x, outputs=y, name='AE'), Model(inputs=x, outputs=h, name='encoder')

class ClusteringLayer(Layer):
    """
    Clustering layer converts input sample (feature) to soft label, i.e. a vector that represents the probability of the
    sample belonging to each cluster. The probability is calculated with student's t-distribution.

    # Example
    ```
        model.add(ClusteringLayer(n_clusters=10))
    ```
    # Arguments
        n_clusters: number of clusters.
        weights: list of Numpy array with shape `(n_clusters, n_features)` witch represents the initial cluster centers.
        alpha: parameter in Student's t-distribution. Default to 1.0.
    # Input shape
        2D tensor with shape: `(n_samples, n_features)`.
    # Output shape
        2D tensor with shape: `(n_samples, n_clusters)`.
    """

    def __init__(self, n_clusters, weights=None, alpha=1.0, **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(ClusteringLayer, self).__init__(**kwargs)
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.initial_weights = weights
        self.input_spec = InputSpec(ndim=2)

    def build(self, input_shape):
        assert len(input_shape) == 2
        input_dim = input_shape[1]
        self.input_spec = InputSpec(dtype=K.floatx(), shape=(None, input_dim))
        self.clusters = self.add_weight(shape=(
            self.n_clusters, input_dim), initializer='glorot_uniform', name='clusters')
        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights
        self.built = True

    def call(self, inputs, **kwargs):
        """ student t-distribution, as same as used in t-SNE algorithm.
                 q_ij = 1/(1+dist(x_i, u_j)^2), then normalize it.
        Arguments:
            inputs: the variable containing data, shape=(n_samples, n_features)
        Return:
            q: student's t-distribution, or soft labels for each sample. shape=(n_samples, n_clusters)
        """
        q = 1.0 / (1.0 + (K.sum(K.square(K.expand_dims(inputs,
                   axis=1) - self.clusters), axis=2) / self.alpha))
        q **= (self.alpha + 1.0) / 2.0
        q = K.transpose(K.transpose(q) / K.sum(q, axis=1))
        return q

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) == 2
        return input_shape[0], self.n_clusters

    def get_config(self):
        config = {'n_clusters': self.n_clusters}
        base_config = super(ClusteringLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class DEC(object):
    def __init__(self,
                 img_size,
                 dims,
                 n_clusters=10,
                 alpha=1.0,
                 init='glorot_uniform'):

        super(DEC, self).__init__()

        self.img_size = img_size

        self.n_clusters = n_clusters
        self.alpha = alpha
        self.autoencoder, self.encoder = autoencoder(self.img_size,dims=dims,  init=init)
        self.autoencoder.summary()
        self.encoder.summary()

        # prepare DEC model
        clustering_layer = ClusteringLayer(self.n_clusters, name='clustering')(layers.Flatten()(self.encoder.output))
        self.model = Model(inputs=self.autoencoder.input, outputs=clustering_layer)

    def pretrain(self, x, y=None, optimizer='adam', epochs=200, batch_size=256, save_dir='results/temp'):
        print('...Pretraining..., batch_size = ', batch_size)
        self.autoencoder.summary()
        self.autoencoder.compile(optimizer=optimizer, loss='mse')
        # self.encoder.summary()

        # logging file
        import csv
        import os
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        csv_logger = callbacks.CSVLogger(save_dir + '/cnnDEC_pretrain_log.csv')
        cb = [csv_logger]
        if y is not None:
            class PrintACC(callbacks.Callback):
                def __init__(self, x, y, ae, enc):
                    self.x = x
                    self.y = y
                    self.autoencoder = ae
                    self.encoder = enc
                    super(PrintACC, self).__init__()

                def on_epoch_end(self, epoch, logs=None):
                    if int(epochs/10) != 0 and epoch % int(epochs/10) != 0:
                        return

                    # features_model = Model(self.autoencoder.input, self.autoencoder.get_layer('hidden_layer').output)                    
                    # print('+ here: ' + 'encoder_%d' % (int(len(self.autoencoder.layers)) - 1))
                    
                    features = layers.Flatten()(self.x)
                    print(features[0])
                    #xx = self.autoencoder.predict(self.x)                    
                    #print(xx[0])
                    #print(self.x[0])
                    # print(np.unique(self.y))
                    # for i in range(0, 10):
                    #     print(self.y[i])
                    # print(self.y)
                    
                    km = KMeans(n_clusters=len(np.unique(self.y)), n_init=20)
                    y_pred = km.fit_predict(features)
                    print(self.y.size, np.unique(self.y), y_pred.size,
                          'encoder_%d' % (int(len(self.model.layers) / 2) - 1))
                    print(' '*8 + '|==>  acc: %.4f,  nmi: %.4f  <==|'
                          % (metrics.acc(self.y, y_pred), metrics.nmi(self.y, y_pred)))

            cb.append(PrintACC(x, y, self.autoencoder, self.encoder))

        # begin pretraining
        t0 = time()
        self.autoencoder.fit(x, x, batch_size=batch_size, epochs=epochs, callbacks=cb)

        print('Pretraining time: %ds' % round(time() - t0))
        self.autoencoder.save_weights(save_dir + '/cnn_ae_weights.h5')
        print('Pretrained weights are saved to %s/cnn_ae_weights.h5' % save_dir)
        self.pretrained = True

    def load_weights(self, weights):  # load weights of DEC model
        self.model.load_weights(weights)

    def extract_features(self, x):
        return self.encoder.predict(x)

    def predict(self, x):  # predict cluster labels using the output of clustering layer
        q = self.model.predict(x, verbose=0)
        return q.argmax(1)

    @staticmethod
    def target_distribution(q):
        weight = q ** 2 / q.sum(0)
        return (weight.T / weight.sum(1)).T

    def compile(self, optimizer='sgd', loss='kld'):
        self.model.compile(optimizer=optimizer, loss=loss)

    def fit(self, x, y=None, maxiter=2e4, batch_size=256, tol=1e-3,
            update_interval=140, save_dir='./results/temp'):

        print('Update interval', update_interval)
        save_interval = int(x.shape[0] / batch_size) * 10  # 10 epochs
        print('Save interval', save_interval)

        # Step 1: initialize cluster centers using k-means
        t1 = time()
        print('Initializing cluster centers with k-means.')
        kmeans = KMeans(n_clusters=self.n_clusters, n_init=20)
        y_pred = kmeans.fit_predict(layers.Flatten()(self.encoder.predict(x)))
        y_pred_last = np.copy(y_pred)
        self.model.get_layer(name='clustering').set_weights(
            [kmeans.cluster_centers_])

        # Step 2: deep clustering
        # logging file
        import csv
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        logfile = open(save_dir + '/dec_log.csv', 'w')
        logwriter = csv.DictWriter(
            logfile, fieldnames=['iter', 'acc', 'nmi', 'ari', 'loss'])
        logwriter.writeheader()

        loss = 0
        index = 0
        index_array = np.arange(x.shape[0])
        for ite in range(int(maxiter)):
            if ite % update_interval == 0:
                q = self.model.predict(x, verbose=0)
                # update the auxiliary target distribution p
                p = self.target_distribution(q)

                # evaluate the clustering performance
                y_pred = q.argmax(1)
                if y is not None:
                    acc = np.round(metrics.acc(y, y_pred), 5)
                    nmi = np.round(metrics.nmi(y, y_pred), 5)
                    ari = np.round(metrics.ari(y, y_pred), 5)
                    loss = np.round(loss, 5)
                    logdict = dict(iter=ite, acc=acc, nmi=nmi,
                                   ari=ari, loss=loss)
                    logwriter.writerow(logdict)
                    print('Iter %d: acc = %.5f, nmi = %.5f, ari = %.5f' % (ite, acc, nmi, ari), ' ; loss=',
                          loss, 'bs=', batch_size, 'max_iter=', maxiter, ', update_interval:', update_interval)

            # train on batch
            if index == 0:
                np.random.shuffle(index_array)
                print('+ index_array:', index_array)
            idx = index_array[index *
                              batch_size: min((index+1) * batch_size, x.shape[0])]
            loss = self.model.train_on_batch(x=x[idx], y=p[idx])
            index = index + 1 if (index + 1) * batch_size <= x.shape[0] else 0

            ite += 1

        # save the trained model
        logfile.close()
        print('saving model to:', save_dir + '/DEC_model_final.h5')
        self.model.save_weights(save_dir + '/DEC_model_final.h5')

        return y_pred


if __name__ == "__main__":
    # setting the hyper parameters
    import argparse

    parser = argparse.ArgumentParser(description='train',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset', default='mnist',
                        choices=['mnist', 'fmnist', 'usps', 'reuters10k', 'stl', 'cifar10'])
    parser.add_argument('--img_size', default=96)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--n_clusters', default=10, type=int)
    parser.add_argument('--maxiter', default=5000, type=int)
    parser.add_argument('--pretrain_epochs', default=None, type=int)
    parser.add_argument('--update_interval', default=None, type=int)
    parser.add_argument('--tol', default=0.001, type=float)
    parser.add_argument('--ae_weights', default=None)
    parser.add_argument('--save_dir', default='results')
    args = parser.parse_args()
    print(args)
    import os
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # load dataset
    from datasets import load_data
    x, y = load_data(args.dataset)
    n_clusters = args.n_clusters
    print('+ load_dataload_dataload_data:', len(x) , len(y), x.size, y.size)

    init = 'glorot_uniform'
    pretrain_optimizer = 'adam'
    bs = 64
    update_interval = 30
    pretrain_epochs = 300
    num_clusters = 10
    clustering_optimizer = 'adam'

    # setting parameters
    if args.dataset == 'mnist' or args.dataset == 'fmnist':
        update_interval = 140
        pretrain_epochs = 300
        init = VarianceScaling(scale=1. / 3., mode='fan_in',
                               distribution='uniform')  # [-limit, limit], limit=sqrt(1./fan_in)
        pretrain_optimizer = SGD(lr=1, momentum=0.9)
        bs = 256
        args.img_size = 28
    elif args.dataset == 'reuters10k':
        update_interval = 30
        pretrain_epochs = 50
        init = VarianceScaling(scale=1. / 3., mode='fan_in',
                               distribution='uniform')  # [-limit, limit], limit=sqrt(1./fan_in)
        pretrain_optimizer = SGD(lr=1, momentum=0.9)
        bs = 128
    elif args.dataset == 'usps':
        update_interval = 30
        pretrain_epochs = 50
        bs = 64
    elif args.dataset == 'stl':
        update_interval = 60
        pretrain_epochs = 100
        bs = 4
        clustering_optimizer = SGD(0.01, 0.9)
        #pretrain_optimizer = SGD(lr=0.1, momentum=0.9)
    elif args.dataset == 'cifar10':
        update_interval = 60
        pretrain_epochs = 30
        bs = 64
        clustering_optimizer = 'adam'
        args.img_size = 32

    # prepare the DEC model
    filters = [64, 96, 128, 128, 64, 32]
    #filters = np.multiply(filters, 3)
    print(filters)
    dec = DEC(img_size=args.img_size, dims=filters,
              n_clusters=num_clusters, init=init)

    path_save = "results/" + args.dataset + "/cnndec"
    print('+ path_save = ', path_save)

    if args.ae_weights is None:
        #dec.autoencoder.load_weights(path_save + '/cnn_ae_weights.h5')
        dec.pretrain(x=x, y=y, optimizer=pretrain_optimizer, epochs=pretrain_epochs, batch_size=bs, save_dir=path_save)
    else:
        dec.autoencoder.load_weights(path_save + '/ae_weights.h5')

    dec.model.summary()
    t0 = time()

    dec.compile(optimizer=clustering_optimizer, loss='kld') 
    y_pred = dec.fit(x, y=y, tol=args.tol, maxiter=args.maxiter,
                     batch_size=bs, update_interval=update_interval, save_dir=path_save)

    print('acc:', metrics.acc(y, y_pred))
    print('clustering time: ', (time() - t0))
