#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 10:40:17 2019

@author: qq
"""
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 指定所用的gpu

# os.environ["CUDA_VISIBLE_DEVICES"] = "" #显存只有8G,当显存不够时，改用CPU跑
import warnings

warnings.filterwarnings("ignore")
import keras.backend.tensorflow_backend as KTF
import tensorflow as tf

#当allow_growth设置为True时，分配器将不会指定所有的GPU内存，而是根据需求增长
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)
KTF.set_session(sess)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import scipy.io as sio
import numpy as np
import argparse
from keras.datasets import mnist, fashion_mnist
from keras.datasets import cifar10

from keras.utils import np_utils
from keras.models import load_model
from LSA_DSA.sa import fetch_dsa, fetch_lsa

from sklearn.model_selection import train_test_split

import time

from FATS.FATS_method import noise_score_var,new_select_from_mmd_Distance_valid,\
    new_select_from_mmd_Distance_test,learning_to_rank_val,learning_to_rank_test

# print(tf.test.is_gpu_available())
# 判断GPU有没有在使用
# physical_devices = tf.config.experimental.list_physical_devices(device_type='GPU')
# assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
# tf.config.experimental.set_memory_growth(physical_devices[0], True)
# 由于在运行多个pycharm时会报内存不足的错误，但内存足够，猜想可能会锁定一部分内存，导致一部分内存不可用，因此添加这部分代码，以解决此问题

CLIP_MAX = 0.5
basedir = os.path.abspath(os.path.dirname(__file__))


## deepgini
def select_deepgini(dataset, model, selectsize, x_test, y_test):
    if dataset == 'mnist' or dataset == 'fashion_mnist':
        x = np.zeros((selectsize, 28, 28, 1))
        y = np.zeros((selectsize,))
    elif dataset == 'cifar10' or dataset == 'svhn':
        x = np.zeros((selectsize, 32, 32, 3))
        y = np.zeros((selectsize,))

    act_layers = model.predict(x_test)
    metrics = np.sum(act_layers ** 2, axis=1)
    rank_lst = np.argsort(metrics)
    for i in range(selectsize):
        x[i] = x_test[rank_lst[i]]
        y[i] = y_test[rank_lst[i]]

    return x, y, rank_lst


def select_rondom(dataset, select_amount, x_test, y_test):
    if dataset == 'mnist' or dataset == 'fashion_mnist':
        x = np.zeros((select_amount, 28, 28, 1))
        y = np.zeros((select_amount,))
    elif dataset == 'cifar10' or dataset == 'svhn':
        x = np.zeros((select_amount, 32, 32, 3))
        y = np.zeros((select_amount,))


    selected_lst = np.random.choice(range(len(x_test)), replace=False, size=len(x_test))

    for i in range(select_amount):
        x[i] = x_test[selected_lst[i]]
        y[i] = y_test[selected_lst[i]]
    return x, y, selected_lst


# 根据indexlist来选择用例
def select_from_index(dataset, select_amount, x_test, indexlst, y_test):
    if dataset == 'mnist' or dataset == 'fashion_mnist':
        x = np.zeros((select_amount, 28, 28, 1))
        y = np.zeros((select_amount,))
    elif dataset == 'cifar10' or dataset == 'svhn':
        x = np.zeros((select_amount, 32, 32, 3))
        y = np.zeros((select_amount,))
    # print(indexlst)
    for i in range(select_amount):
        x[i] = x_test[indexlst[i]]
        y[i] = y_test[indexlst[i]]
    return x, y

# 找出max_lsa在 target_lsa中的index，排除selected_lst中已经选的
def find_index(target_lsa, selected_lst, max_lsa):
    for i in range(len(target_lsa)):
        if max_lsa == target_lsa[i] and i not in selected_lst:
            return i
    return 0


# 重新修改
def order_output(target_lsa, select_amount):
    lsa_lst = []

    tmp_lsa_lst = target_lsa[:]
    selected_lst = []
    while len(selected_lst) < select_amount:
        max_lsa = max(tmp_lsa_lst)
        selected_lst.append(find_index(target_lsa, selected_lst, max_lsa))
        lsa_lst.append(max_lsa)
        tmp_lsa_lst.remove(max_lsa)
    return selected_lst, lsa_lst


def fetch_our_measure(model, x_test):
    bound_data_lst = []
    # x_test=x_test.astype('float32').reshape(-1,28,28,1)
    # x_test/=255
    act_layers = model.predict(x_test)

    ratio_lst = []
    for i in range(len(act_layers)):
        act = act_layers[i]
        _, __, ratio = find_second(act)
        ratio_lst.append(ratio)

    return ratio_lst


def find_second(act):
    max_ = 0
    second_max = 0
    sec_index = 0
    max_index = 0
    for i in range(10):
        if act[i] > max_:
            max_ = act[i]
            max_index = i

    for i in range(10):
        if i == max_index:
            continue
        if act[i] > second_max:  # 第2大加一个限制条件，那就是不能和max_一样
            second_max = act[i]
            sec_index = i
    ratio = 1.0 * second_max / max_
    # print 'max:',max_index
    return max_index, sec_index, ratio  # ratio是第二大输出达到最大输出的百分比


# 输入第一第二大的字典，输出selected_lst。用例的index
def select_from_firstsec_dic(selectsize, dicratio, dicindex):
    selected_lst = []
    tmpsize = selectsize
    # tmpsize保存的是采样大小，全程都不会变化

    noempty = no_empty_number(dicratio)
    print(selectsize)
    print(noempty)
    # 待选择的数目大于非空的类别数(满载90类)，每一个都选一个
    while selectsize >= noempty:
        for i in range(100):
            if len(dicratio[i]) != 0:  # 非空就选一个最大的出来
                tmp = max(dicratio[i])
                j = dicratio[i].index(tmp)
                # if tmp>=0.1:
                selected_lst.append(dicindex[i][j])
                dicratio[i].remove(tmp)
                dicindex[i].remove(dicindex[i][j])
        selectsize = tmpsize - len(selected_lst)
        noempty = no_empty_number(dicratio)
        # print(selectsize)
    # selectsize<noempty
    # no_empty_number(dicratio)
    print(selectsize)

    # 剩下少量样本没有采样，比如还存在30类别非空，但是只要采样10个，此时我们取30个最大值中的前10大
    while len(selected_lst) != tmpsize:
        max_tmp = [0 for i in range(selectsize)]  # 剩下多少就申请多少
        max_index_tmp = [0 for i in range(selectsize)]
        for i in range(100):
            if len(dicratio[i]) != 0:
                tmp_max = max(dicratio[i])
                if tmp_max > min(max_tmp):
                    index = max_tmp.index(min(max_tmp))
                    max_tmp[index] = tmp_max
                    # selected_lst.append()
                    # if tmp_max>=0.1:
                    max_index_tmp[index] = dicindex[i][dicratio[i].index(tmp_max)]  # 把样本序列号存在此列表中
        if len(max_index_tmp) == 0 and len(selected_lst) != tmpsize:
            print('wrong!!!!!!')
            break
        selected_lst = selected_lst + max_index_tmp
        # print(len(selected_lst))
    # print(selected_lst)
    assert len(selected_lst) == tmpsize
    return selected_lst  # 返回筛选出来的测试样本


# 配对表情非空的数目。比如第一是3，第二是5，此时里面没有任何实例存在那么就是0
def no_empty_number(dicratio):
    no_empty = 0
    for i in range(len(dicratio)):
        if len(dicratio[i]) != 0:
            no_empty += 1
    return no_empty  #返回存在实例的第一大类和第二大类组合个数


# 返回MCP优化版的采样用例
def select_mcp(dataset, model, selectsize, x_test, y_test):
    if dataset == 'mnist' or dataset == 'fashion_mnist':
        x = np.zeros((selectsize, 28, 28, 1))
        y = np.zeros((selectsize,))
    elif dataset == 'cifar10' or dataset == 'svhn':
        x = np.zeros((selectsize, 32, 32, 3))
        y = np.zeros((selectsize,))
    act_layers = model.predict(x_test)
    dicratio = [[] for i in range(100)]  # 只用90，闲置10个
    dicindex = [[] for i in range(100)]
    for i in range(len(act_layers)):
        act = act_layers[i]
        max_index, sec_index, ratio = find_second(act)  # max_index
        # 按照第一和第二大的标签来存储，比如第一是8，第二是4，那么就存在84里，比例和测试用例的序号

        dicratio[max_index * 10 + sec_index].append(ratio)
        dicindex[max_index * 10 + sec_index].append(i)


    selected_lst = select_from_firstsec_dic(selectsize, dicratio, dicindex)
    for i in range(selectsize):
        x[i] = x_test[selected_lst[i]]
        y[i] = y_test[selected_lst[i]]

    return x, y, selected_lst


# 返回DSA意外性分数所选出的样本
def select_from_large(dataset, select_amount, x_test, target_lsa, y_test):
    if dataset == 'mnist' or dataset == 'fashion_mnist':
        x = np.zeros((select_amount, 28, 28, 1))
        y = np.zeros((select_amount,))
    elif dataset == 'cifar10' or dataset == 'svhn':
        x = np.zeros((select_amount, 32, 32, 3))
        y = np.zeros((select_amount,))
    selected_lst, lsa_lst = order_output(target_lsa, select_amount)
    # selected_lst, lsa_lst = order_output(target_lsa, len(x_test))
    # print(lsa_lst)
    # print("根据DSA分数筛选出来的样本所在下标selected_lst"+str(selected_lst))
    # print("根据DSA分数筛选出来的样本lsa_lst" + str(lsa_lst))

    for i in range(select_amount):
        x[i] = x_test[selected_lst[i]]
        y[i] = y_test[selected_lst[i]]
    return x, y, selected_lst


# 输入：待测模型、排序度量lsa还是dsa,待测试的集合，选择的size
# 输出：准确率，在待测试的集合上
# 把预测的标签作为伪标签
def datapredict(dataset='mnist', attack='fgsm'):
    if dataset == 'mnist':
        # train_images 和 train_labels 数组是训练集，即模型用于学习的数据。
        # 测试集、test_images 和 test_labels 数组会被用来对模型进行测试。
        # 将这些值缩小至 0 到 1 之间，然后将其馈送到神经网络模型。为此，请将这些值除以 255。请务必以相同的方式对训练集和测试集进行预处理：

        (x_train, y_train), (x_test, y_test) = mnist.load_data()  # 60000张的训练图像和10000张的测试图像
        # 划分训练集和验证集
        x_test = x_test.astype("float32").reshape(-1, 28, 28, 1)
        x_test = x_test / 255.0

        x_train = x_train.astype("float32").reshape(-1, 28, 28, 1)

        # x_train = (x_train / 255.0) - (1.0 - CLIP_MAX)
        x_train = x_train / 255.0

        # 将训练集划分为训练集和验证集，训练集为48000，验证集为12000， #测试集为10000
        x_train_spilt, x_valid, y_train_spilt, y_valid = train_test_split(x_train, y_train, test_size=0.024,
                                                                          shuffle=True)
        print("x_train" + str(type(x_train)) + str(x_train.shape))
        if attack !='none':

            npzfile = np.load('./dataset/mnist/adv/mnist_' + attack + '_compound8.npz')

            x_test = npzfile['x_test']
            y_test = npzfile['y_test']
        # print("x_test1" + str(type(x_test)) + str(x_test.shape))
        # print("x_valid" + str(type(x_valid)) + str(x_valid.shape))
        print("y_valid" + str(type(y_valid)) + str(y_valid.shape))
        print("y_test" + str(type(y_test)) + str(y_test.shape))


        valid_len = len(x_valid)

    elif dataset == 'fashion_mnist':

        (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()  # 60000张的训练图像和10000张的测试图像
        # 划分训练集和验证集
        x_test = x_test.astype("float32").reshape(-1, 28, 28, 1)
        x_test = x_test / 255.0

        x_train = x_train.astype("float32").reshape(-1, 28, 28, 1)

        # x_train = (x_train / 255.0) - (1.0 - CLIP_MAX)
        x_train = x_train / 255.0

        # 将训练集划分为训练集和验证集，训练集为48000，验证集为12000， #测试集为10000
        x_train_spilt, x_valid, y_train_spilt, y_valid = train_test_split(x_train, y_train, test_size=0.024,
                                                                          shuffle=True)
        print("x_train" + str(type(x_train)) + str(x_train.shape))

        if attack != 'none':
            npzfile = np.load('./dataset/fashion_mnist/adv/fashion_mnist_' + attack + '_compound11.npz')
            x_test = npzfile['x_test']
            y_test = npzfile['y_test']

        print("y_valid" + str(type(y_valid)) + str(y_valid.shape))
        print("y_test" + str(type(y_test)) + str(y_test.shape))

        valid_len = len(x_valid)

    elif dataset == 'cifar10':

        (x_train, y_train), (x_test, y_test) = cifar10.load_data()  # 50000条训练数据.10000条测试数据
        x_train = x_train.astype("float32").reshape(-1, 32, 32, 3)
        x_train = x_train / 255.0

        x_test = x_test.astype("float32")
        x_test = x_test / 255.0

        # 将训练集划分为训练集和验证集，训练集为38000，验证集为12000， #测试集为10000
        x_train_spilt, x_valid, y_train_spilt, y_valid = train_test_split(x_train, y_train, test_size=0.03,
                                                                          shuffle=True)
        print("x_train" + str(type(x_train)) + str(x_train.shape))
        if attack != 'none':
            npzfile = np.load('./dataset/cifar10/adv/cifar10_' + attack + '_compound9.npz')
            x_test = npzfile['x_test']
            y_test = npzfile['y_test']

        y_valid = y_valid.reshape(-1)

        print("y_valid" + str(type(y_valid)) + str(y_valid.shape))
        print("y_test" + str(type(y_test)) + str(y_test.shape))

        valid_len = len(x_valid)
    elif dataset == 'svhn':  # 训练集：73257张图像；测试集：26032张图像。
        train = sio.loadmat("dataset/svhn/svhn_data/svhn_train.mat")
        x_train = np.transpose(train['X'], axes=[3, 0, 1, 2])
        x_train = x_train.astype('float32') / 255.0
        y_train = np.reshape(train['y'], (-1,)) - 1

        test = sio.loadmat("dataset/svhn/svhn_data/svhn_test.mat")
        x_test = np.transpose(test['X'], axes=[3, 0, 1, 2])
        x_test = x_test.astype('float32') / 255.0
        y_test = np.reshape(test['y'], (-1,)) - 1

        # 将训练集划分为训练集和验证集，#训练集：73257张图像，验证集为12000， #测试集：26032张图像。
        x_train_spilt, x_valid, y_train_spilt, y_valid = train_test_split(x_train, y_train, test_size=0.03,
                                                                          shuffle=True)
        print("x_train" + str(type(x_train)) + str(x_train.shape))

        if attack != 'none':
            npzfile = np.load('./dataset/svhn/adv/svhn_' + attack + '_compound10.npz')
            x_test = npzfile['x_test']
            y_test = npzfile['y_test']

        y_valid = y_valid.reshape(-1)

        print("y_valid" + str(type(y_valid)) + str(y_valid.shape))
        print("y_test" + str(type(y_test)) + str(y_test.shape))

        valid_len = len(x_valid)
    return x_train, x_test, y_test, x_valid, y_valid, valid_len


def retrain(begin,res, train_act_layers, model, args, layer_names,ranknet_model,x_train, x_test, y_test, selectsize=100,
            attack='fgsm', measure='MCP',flag=1):

    # begin = time.time() #计算每个测试排序方法的时间成本：开始时间
    target_lst = []
    # baselines =['LSA','DSA','CES','MCP','SRS','AAL','FATS','random-Distance','deepgini']
    if measure == 'FATS':
        test_feature = new_select_from_mmd_Distance_test(res, train_act_layers, model, x_test, flag)
        x_select, y_select, rank_list = learning_to_rank_test(args.d, ranknet_model,selectsize, x_test, y_test, test_feature)  # 对测试样本进行打分排序

    if measure == 'deepgini':
        x_select, y_select, rank_list = select_deepgini(args.d, model, selectsize, x_test, y_test)

    if measure == 'SRS':
        x_select, y_select, rank_list = select_rondom(args.d, selectsize, x_test, y_test)
    if measure == "MCP":
        x_select, y_select, rank_list = select_mcp(args.d, model, selectsize, x_test, y_test)
    if measure == 'LSA':
        target_lst = fetch_lsa(model, x_train, x_test, attack, layer_names, args)  # 衡量相较于训练样本的多样性
    if measure == 'DSA':
        target_lst = fetch_dsa(model, x_train, x_test, attack, layer_names, args)  # 衡量相较于训练样本的多样性

    elif measure not in ['SRS', 'MCP', 'FATS',  'deepgini']:
        x_select, y_select, rank_list = select_from_large(args.d, selectsize, x_test, target_lst, y_test)


    end = time.time()  # 计算每个测试排序方法的时间成本:结束时间
    Time = end - begin # 计算每个测试排序方法总的时间开销

    y_select = np_utils.to_categorical(y_select, 10)
    y_test = np_utils.to_categorical(y_test, 10)

    score = model.evaluate(x_test, y_test, verbose=0)
    # print('Test Loss: %.4f' % score[0])
    print('Before retrain, Test accuracy: %.4f' % score[1])
    origin_acc = score[1]

    # ---------------------------------模型重训练-----------------------------
    model.compile(loss="categorical_crossentropy", optimizer="adadelta", metrics=['accuracy'])
    # retrain_acc = 0

    model.fit(x_select, y_select, batch_size=100, epochs=5, shuffle=True, verbose=1,
              validation_data=(x_test, y_test))  # 模型重训练
    score = model.evaluate(x_test, y_test, verbose=0)
    retrain_acc = score[1]

    return  Time,x_select, y_select, retrain_acc, origin_acc, rank_list

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--d", "-d", help="Dataset", type=str, default="cifar10")
    parser.add_argument("--exp_id", "-exp_id", help="model", type=str)
    parser.add_argument(
        "--target",
        "-target",
        help="Target input set (test or adversarial set)",
        type=str,
        default="fgsm",
    )
    parser.add_argument(
        "--save_path", "-save_path", help="Save path", type=str, default="./tmp/"
    )
    parser.add_argument(
        "--batch_size", "-batch_size", help="Batch size", type=int, default=128
    )
    parser.add_argument(
        "--var_threshold",
        "-var_threshold",
        help="Variance threshold",
        type=int,
        default=1e-5,
    )
    parser.add_argument(
        "--upper_bound", "-upper_bound", help="Upper bound", type=int, default=2000
    )
    parser.add_argument(
        "--n_bucket",
        "-n_bucket",
        help="The number of buckets for coverage",
        type=int,
        default=1000,
    )
    parser.add_argument(
        "--num_classes",
        "-num_classes",
        help="The number of classes",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--is_classification",
        "-is_classification",
        help="Is classification task",
        type=bool,
        default=True,
    )
    args = parser.parse_args()
    print(args.exp_id)

    if args.d == 'mnist' or args.d == 'fashion_mnist':
        layer_names = ['activation_1']  # 需要更改，用于计算DSA和LSA
    elif args.d == 'cifar10':
        layer_names = ['batch_normalization_39']
    elif args.d == 'svhn':
        layer_names = ['batch_normalization_16']  #batch_normalization_16

    if args.d == 'svhn':
        selectsizelist = [260, 780, 1300, 2080, 2600]
    else:
        selectsizelist = [100, 300, 500, 800, 1000]

    # baselines = ['FATS','MCP','SRS','DeepGini','LSA','DSA']
    baselines = ['FATS']
    operators = ['fgsm']
    # operators = ['fgsm', 'jsma','bim-a', 'bim-b', 'cw-l2']

    model_path = 'model/model_' + args.d + '.hdf5'
    model = load_model(model_path)
    print(model.summary())

    res = []
    train_act_layers = []
    ranknet_model = None
    valid_flag = 0

    for operator in operators:
        x_train, x_test, y_test, x_valid, y_valid, valid_len = datapredict(args.d, operator)
        for measure in baselines:
            if measure == 'FATS' and valid_flag == 0:
                print("---------排序模型构建---------------")
                begin = time.time()  # 计算每个测试排序方法的时间成本：开始时间
                valid_noise = noise_score_var(model, args.d, x_valid, y_valid)
                valid_feature, res, train_act_layers = new_select_from_mmd_Distance_valid(model, x_valid, x_train,
                                                                                          valid_flag)
                ranknet_model = learning_to_rank_val(valid_feature, valid_noise)
                valid_flag = 1

            resultfilename = './result/' + args.d + '_compound_' + operator + '.txt'
            result_to_write = ''
            result_to_write += measure + ':\n'


            origin_acc = 0

            print("操作类型" + str(operator))
            print("筛选方法" + str(measure))

            for selectsize in selectsizelist:

                Time,x_select,y_select,retrain_acc, origin_acc, rank_list = retrain(begin,res, train_act_layers,model, args, layer_names,ranknet_model, x_train, x_test,
                                                             y_test, selectsize, operator, measure,valid_flag)


                model = load_model(model_path)

                print("用时为:" + str(Time))
                result_to_write += 'selectsize:' + str(selectsize) + ','
                result_to_write += 'Time:' + str(round(Time, 4)) + 's,'

                result_to_write += 'original acc: ' + str(round(origin_acc, 4)) + ','
                result_to_write += 'retrain_acc:' + str(round(retrain_acc, 4)) + '\n'

                with open(resultfilename, 'a') as file_object:
                    file_object.write(result_to_write)

