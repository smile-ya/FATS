from __future__ import absolute_import
from __future__ import print_function

import argparse
from util import get_data, get_model
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model

from keras.utils import multi_gpu_model
from sklearn.model_selection import train_test_split
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"


def train(dataset='mnist', batch_size=128, epochs=50, pretrained=False):
    """
    Train one model with data augmentation: random padding+cropping and horizontal flip
    :param args: 
    :return: 
    """

    print('Data set: %s' % dataset)
    X_train, Y_train, X_test, Y_test = get_data(dataset)

    # #  划分训练集和验证集
    # X_train, X_valid, Y_train, Y_valid = train_test_split(X_train, Y_train, test_size=0.2,
    #                                                       shuffle=True)
    acc = 0
    if pretrained == True:
        model = load_model('../model/model_%s.hdf5' % dataset)
        score = model.evaluate(X_test, Y_test, verbose=0)
        acc = score[1]
    else:
        model = get_model(dataset)
    # model = multi_gpu_model(model, gpus=4)  # 使用多gpu进行训练
    model.compile(
        loss='categorical_crossentropy',
        optimizer='adadelta',
        metrics=['accuracy']
    )

    # training with data augmentation
    # data augmentation 训练数据增强
    datagen = ImageDataGenerator(
        rotation_range=20)

    model.fit_generator(
        datagen.flow(X_train, Y_train, batch_size=batch_size),
        steps_per_epoch=len(X_train) / batch_size,
        epochs=epochs,
        verbose=1,
        validation_data=(X_test, Y_test))

    print("##############################训练结束###################################")
    if pretrained == True:
        score = model.evaluate(X_test, Y_test, verbose=0)
        if score[1] > acc:
            print("model refined!")
            model.save('../model/model_%s.hdf5' % dataset)
    else:
        model.save('../model/model_%s.hdf5' % dataset)


def main(args):
    """
    Train model with data augmentation: random padding+cropping and horizontal flip
    :param args:
    :return:
    """
    assert args.dataset in ['mnist', 'cifar10', 'svhn', 'cifar100', 'fashion-mnist', 'all'], \
        "dataset parameter must be either 'mnist', 'cifar10', 'svhn', 'cifar100', 'fashion-mnist', 'all'"
    if args.dataset == 'all':
        for dataset in ['mnist', 'cifar10', 'svhn', 'cifar100', 'fashion-mnist']:
            train(dataset, args.batch_size, args.epochs)
    else:
        train(args.dataset, args.batch_size, args.epochs, args.pretrained)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-d', '--dataset',
        help="Dataset to use; either 'mnist', 'cifar10', 'svhn','cifar100','fashion-mnist' or 'all'",
        required=False, type=str,  # required - 是否必须要提供子命令，默认为 False
    )
    parser.add_argument(
        '-e', '--epochs',
        help="The number of epochs to train for.",
        required=False, type=int,
    )
    parser.add_argument(
        '-b', '--batch_size',
        help="The batch size to use for training.",
        required=False, type=int,
    )
    parser.add_argument(
        '-p', '--pretrained',
        help="Where train from a pretrained model.",
        required=False, type=bool
    )
    parser.set_defaults(dataset="mnist")
    parser.set_defaults(epochs=120)
    parser.set_defaults(batch_size=100)
    parser.set_defaults(pretrained=False)
    args = parser.parse_args()
    main(args)
