#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import argparse
import os
import sys

import keras
from keras import Model,Input
from keras.models import load_model
from keras.layers import Activation,Flatten
import math
import numpy as np
import pandas as pd
import foolbox
from tqdm import tqdm
from keras.datasets import mnist
from keras.datasets import cifar10
from dataset.util import get_data
import scipy.io as sio

sys.path.append(os.path.dirname(__file__) + os.sep + '../../')

import warnings
warnings.filterwarnings("ignore")
import multiprocessing

def adv_func(x,y,model_path,dataset='mnist',attack='fgsm'):
    keras.backend.set_learning_phase(0)
    model=load_model(model_path)
    foolmodel=foolbox.models.KerasModel(model,bounds=(0,1),preprocessing=(0,1))
    if attack=='cw':
        #attack=foolbox.attacks.IterativeGradientAttack(foolmodel)
        attack=foolbox.attacks.L2BasicIterativeAttack(foolmodel)
    elif attack=='fgsm':
        # FGSM
        attack=foolbox.attacks.GradientSignAttack(foolmodel)
    elif attack=='bim':
        # BIM
        attack=foolbox.attacks.L1BasicIterativeAttack(foolmodel)
    elif attack=='jsma':
        # JSMA
        attack=foolbox.attacks.SaliencyMapAttack(foolmodel)
        # CW
        #attack=foolbox.attacks.DeepFoolL2Attack(foolmodel)
    result=[]
    if dataset=='mnist':
        w,h=28,28
    elif dataset=='cifar10':
        w,h=32,32
    else:
        return False
    test_index = -1
    for image in tqdm(x):
        try:
            #adv=attack(image.reshape(28,28,-1),label=y,steps=1000,subsample=10)
            #adv=attack(image.reshape(w,h,-1),y,epsilons=[0.01,0.1],steps=10)
            test_index += 1
            if attack!='fgsm':
                adv=attack(image.reshape(w,h,-1),y[test_index])
                adv=attack(image.reshape(w,h,-1),y[test_index])
                adv=attack(image.reshape(w,h,-1),y[test_index])
            else:
                adv=attack(image.reshape(w,h,-1),y[test_index],[0.01,0.1])

            if isinstance(adv,np.ndarray):
                result.append(adv)
            else:
                print('adv fail')
        except:
            pass
    return np.array(result)

def generate_mnist_sample(attack):
    (X_train, Y_train), (X_test, Y_test) = get_data('mnist') # 28*28

    image_org=X_test
    adv=adv_func(image_org,Y_test,model_path='./model/model_mnist.hdf5',dataset='mnist',attack=attack)
    return adv

def generate_cifar10_sample(attack):
    (X_train, Y_train), (X_test, Y_test) = cifar10.load_data()  # 32*32
    X_train = X_train.astype('float32').reshape(-1,32,32,3)
    X_test = X_test.astype('float32').reshape(-1,32,32,3)
    X_train /= 255
    X_test /= 255

    Y_train=Y_train.reshape(-1)
    Y_test=Y_test.reshape(-1)

    image_org=X_test

    adv=adv_func(image_org,Y_test,model_path='../../model/model_cifar10.hdf5',dataset='cifar10',attack=attack)
    return adv

def generate_svhn_sample(label,attack):
    train = sio.loadmat("./svhn/svhn_data/svhn_train.mat")
    test = sio.loadmat("./svhn/svhn_data/svhn_test.mat")

    X_train = np.transpose(train['X'], axes=[3, 0, 1, 2])
    X_train = X_train.astype('float32') / 255.0

    X_test = np.transpose(test['X'], axes=[3, 0, 1, 2])
    X_test = X_test.astype('float32') / 255.0

    Y_train = np.reshape(train['y'], (-1,)) - 1
    Y_test = np.reshape(test['y'], (-1,)) - 1

    image_org=X_test[np.argmax(Y_test,axis=1)==label]

    adv=adv_func(image_org,label,model_path='../../model/model_svhn.hdf5',dataset='cifar10',attack=attack)
    return adv

def generate_adv_sample(dataset,attack):
    if dataset=='mnist':
        sample_func=generate_mnist_sample
    elif dataset=='svhn':
        sample_func=generate_svhn_sample
    elif dataset=='cifar10':
        sample_func=generate_cifar10_sample
    else:
        print('erro')
        return
    X_adv=sample_func(attack=attack)
    return X_adv


