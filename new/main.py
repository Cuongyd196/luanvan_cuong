
import tensorflow as tf

physical_devices = tf.config.list_physical_devices('GPU') 
tf.config.experimental.set_memory_growth(physical_devices[0], True)

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score
import tensorflow as tf
import keras
from ultils import cluster_acc , set_memory_growth
import argparse


def main(x,y):
    # clustering
    print('Clustering.....')
    km = KMeans(n_clusters=len(np.unique(y)), n_init=20)
    y_pred = km.fit_predict(x)

    # evaluate
    print('Result:')
    acc = cluster_acc(y, y_pred)
    nmi = normalized_mutual_info_score(y, y_pred)
    print('acc:', acc)
    print('nmi:', nmi)

if __name__ == '__main__':
    from datasets import load_data

    # setting the hyper parameters
    parser = argparse.ArgumentParser(description='train',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset', default='mnist',
                        choices=['mnist', 'fmnist', 'usps', 'reuters10k', 'stl', 'cifar10',  'mnist_vgg16', 'cifar10_vgg16'])
    args = parser.parse_args()
    print(args)
    x, y = load_data(args.dataset)
    print('+ Load_data - len(x) , len(y), x.size, y.size :', len(x) , len(y), x.size, y.size)

    main(x,y)

