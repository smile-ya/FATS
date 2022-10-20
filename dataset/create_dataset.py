#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 10:40:17 2019

@author: qq
"""
import os

from keras.utils import np_utils

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = "2" #显存只有8G,当显存不够时，改用CPU跑
import warnings
warnings.filterwarnings("ignore")
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import numpy as np
import argparse
import scipy.io as sio
from keras.datasets import mnist,cifar10,fashion_mnist
from keras.models import Model, Input, load_model
CLIP_MAX = 0.5

def createdataset(dataset , attack, ratio=8):

    x_target = np.load('./mutation/adv/' + dataset + '/Adv_' + dataset + '_' + attack + '.npy')

    if dataset == 'mnist' :
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_test = x_test.astype("float32").reshape(-1, 28, 28, 1)
        x_test = x_test / 255.0
        origin_lst = np.random.choice(range(10000), replace=False, size=ratio * 1000)
        mutated_lst = np.random.choice(range(10000), replace=False, size=10000 - ratio * 1000)

        x_dest = np.append(x_test[origin_lst], x_target[mutated_lst], axis=0)
        y_dest = np.append(y_test[origin_lst], y_test[mutated_lst])
        np.savez('./mnist/adv/mnist_' + attack + '_compound8.npz', x_test=x_dest, y_test=y_dest)

    elif dataset == 'fashion_mnist':
        (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
        x_test = x_test.astype("float32").reshape(-1, 28, 28, 1)
        x_test = x_test / 255.0

        origin_lst = np.random.choice(range(10000), replace=False, size=ratio * 1000)
        mutated_lst = np.random.choice(range(10000), replace=False, size=10000 - ratio * 1000)

        x_dest = np.append(x_test[origin_lst], x_target[mutated_lst], axis=0)
        y_dest = np.append(y_test[origin_lst], y_test[mutated_lst])
        np.savez('./fashion_mnist/adv/fashion_mnist_' + attack + '_compound11.npz', x_test=x_dest, y_test=y_dest)

    elif dataset == 'cifar10' :
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()

        x_test = x_test.astype("float32")
        x_test = x_test / 255.0

        origin_lst = np.random.choice(range(10000), replace=False, size=ratio * 1000)
        mutated_lst = np.random.choice(range(10000), replace=False, size=10000 - ratio * 1000)

        x_dest = np.append(x_test[origin_lst], x_target[mutated_lst], axis=0)
        y_dest = np.append(y_test[origin_lst], y_test[mutated_lst])
        if attack in ['rotation', 'translation', 'shear', 'brightness', 'contrast', 'scale']:
            np.savez('./cifar10/imagetrans/cifar10_' + attack + '_compound9.npz', x_test=x_dest, y_test=y_dest)
        else:
            np.savez('./cifar10/adv/cifar10_' + attack + '_compound9.npz', x_test=x_dest, y_test=y_dest)
    elif dataset == 'svhn':                                              #训练集：73257张图像；测试集：26032张图像。
        train = sio.loadmat("./svhn/svhn_data/svhn_train.mat")
        test =  sio.loadmat("./svhn/svhn_data/svhn_test.mat")

        x_train = np.transpose(train['X'], axes=[3, 0, 1, 2])
        x_train = x_train.astype('float32') / 255.0

        x_test = np.transpose(test['X'], axes=[3, 0, 1, 2])
        x_test = x_test.astype('float32') / 255.0

        y_train = np.reshape(train['y'], (-1,)) - 1
        y_test = np.reshape(test['y'], (-1,)) - 1

        origin_lst = np.random.choice(range(26032), replace=False, size=int(0.75 * 26032))    #75%的原始数据
        mutated_lst = np.random.choice(range(26032), replace=False, size=int(26032 - 0.75 * 26032))

        x_dest = np.append(x_test[origin_lst], x_target[mutated_lst], axis=0)
        y_dest = np.append(y_test[origin_lst], y_test[mutated_lst])

        np.savez('./svhn/adv/svhn_' + attack + '_compound10.npz', x_test=x_dest, y_test=y_dest)

    y_dest = np_utils.to_categorical(y_dest, 10)
    model = load_model('../model/model_%s.hdf5' % dataset)
    score = model.evaluate(x_dest, y_dest, verbose=0)
    acc = score[1]
    print("%s After %s, testing Acc:%s" % (dataset, attack, acc))

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--d", "-d", help="Dataset", type=str, default="fashion_mnist")
    parser.add_argument(
        "--target",
        "-target",
        help="Target input set (test or adversarial set)",
        type=str,
        default="fgsm",
    )
    args = parser.parse_args()
    createdataset(args.d , args.target, ratio=8)