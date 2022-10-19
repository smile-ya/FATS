import os
import math
# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
# os.environ["CUDA_VISIBLE_DEVICES"] = "3" #指定所用的gpu

import warnings
warnings.filterwarnings("ignore")

import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

import scipy
import numpy as np
import xgboost as xgb
import hdbscan
import math
import random
from xgboost import DMatrix
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import FastICA
from mmd_critic.run_digits_new import run
from sklearn.cluster import KMeans

basedir = os.path.abspath(os.path.dirname(__file__))

def noise_score_var(model, dataset, x_target,y_target):
    act_layers = model.predict(x_target)
    # print("y_traget")
    # print(np.full(act_layers
    #     # print(y_target)[0].shape, act_layers[0][y_target[0]]))
    # print(act_layers[0].tolist()[y_target[0]])
    varnoise = []
    for i in range(len(act_layers)):
        act = act_layers[i]
        target_class_prob = np.full(act.shape,act[y_target[i]])
        # act_var = np.linalg.norm(act-target_class_prob)
        act_var = -math.pow(np.linalg.norm(act - target_class_prob), 2)
        varnoise.append(act_var)
    print("######################valid_varnoise####################")

    varnoise_list = []
    for i in range(len(varnoise)):
        for j in range(len(varnoise)):
            if i == j:
                continue
            if varnoise[i] <  varnoise[j]:
                varnoise_list.append(1)
            if varnoise[i] == varnoise[j]:
                varnoise_list.append(0)
            if varnoise[i] > varnoise[j]:
                varnoise_list.append(-1)
    print("############################varnoise_list#########################3")
    # print(len(varnoise_list))
    return varnoise_list

def class_cluster_kmeans(model, x_train):
    # 实例化K-Means算法模型，定义k为10
    cluster = KMeans(n_clusters=10, random_state=0)
    train_act_layers = model.predict(x_train)
    #
    # minMax = MinMaxScaler()
    # train_act_layers = minMax.fit_transform(train_act_layers)  # 对倒数第三层flatten层上的输出向量进行归一化

    print("-----------------act_layers-----------------")
    print(train_act_layers.shape)

    # 使用数据集X进行训练

    r = cluster.fit(train_act_layers)
    # 调用属性labels_，查看聚类结果
    labels = r.labels_
    y_pred_list = labels.tolist()
    countlist = []
    print("------labels------")
    print(labels)

    for i in range(np.min(labels), np.max(labels) + 1):
        print(i)
        countlist.append(y_pred_list.count(i))  # 计算每个集群的样本数目，
    print("集群样本数目列表为" + str(countlist))  # 返回集群样本数目列表
    sum = 0
    for l in countlist:
        sum += l
    print("所有集群样本数目总和" + str(sum))

    # print(np.sort(countlist)) #按照集群数目从小到大排序
    # print(np.argsort(countlist)) #返回排序后的下标

    label_noise = []
    for i, l in enumerate(labels):
        if l == -1:
            label_noise.append(i)  # 将数据异常点存储到label_noise中

    res = {}
    for i, l in enumerate(labels):
        if l != -1:
            if l not in res:
                res[l] = []
            res[l].append(i)  # 将0，1，2三个集群下的测试样本的下标存储到res[0],res[1],res[2]

    # for key in res:
    #     print("-----------------------res[key]---------------------")
    #     print(key)
    #     print(len(res[key]))

    print("-----------------------将成二维后的act_layers---------------")
    print(train_act_layers.shape)
    print(len(res))
    # print(act_layers)
    return res, train_act_layers, countlist

# 使用MMD-critic 方法从每个聚类中选择5个聚类质心，然后直接计算所有测试样本与这些聚类质心的距离。
def mmd_critic_centroid(res,train_act_layers):

    # minMax = MinMaxScaler()
    # test_act_layers = minMax.fit_transform(test_act_layers)  # 对倒数第三层flatten层上的输出向量进行归一化
    # dec_dim = 2
    # fica = FastICA(n_components=dec_dim)  # 当特征维度增加时，hdbscan性能可能会大大降低，因此在聚类前使用FastICA算法进行降维
    # test_act_layers = fica.fit_transform(test_act_layers)  # 得到降维后的特征向量


    selectlist = {}
    # 非异常点每一类的排序，key类别号
    res_get_s = {}  # 存储每个聚类中具有代表性的数据的下标，下标是对应的集群的数据的下标，格式为{0：[0,1,2,3],1:[1,2,3,...]}
    for key in res:
        temp_dense = []
        for l in res[key]:
            temp_dense.append(train_act_layers[l])
        temp_dense = np.array(temp_dense)
        temp_label = np.full((len(temp_dense)), key)
        mmd_res, _ = run(temp_dense, temp_label, gamma=0.026, m=min(5, len(temp_dense)), k=0, ktype=0, outfig=None,
                         critoutfig=None,
                         testfile=os.path.join(basedir,
                                               '../mmd_critic/data/a.txt'))  # 使用MMD-critic算法（可用于机器学习模型的可解释性）从一组聚类中选择最具代笔性的输入,从每个集群中选择几个具有代表性的数据
        res_get_s[key] = mmd_res

    for key in res:
        selectlist[key] = []
        for i in range(len(res_get_s[key])):
            selectlist[key].append(res[key][res_get_s[key][i]])
    return selectlist

# 计算所有未选中的测试样本与每个类别中m个测试样本的平均距离
def maxdistance_new(selectlist, train_act_layers, model,x_test):  # selectlist:存储每个集群中每个子区间中被选择的样本，格式为：{0：[23,45,433,56],1:[32,45,66],2[3221,33,11]}
    test_act_layers = model.predict(x_test)
    avglist = {}
    for key in selectlist:
        avglist[key] = []

        if len(selectlist[key]) == 0:
            continue
        for i in range(len(test_act_layers)):
            d = 0
            for j in range(len(selectlist[key])):
                d += np.linalg.norm(test_act_layers[i] - train_act_layers[selectlist[key][j]])
            avg_d = d / len(selectlist[key])
            avglist[key].append(avg_d)  # 存储每个测试样本与每个集群中5个已选择测试样本之间的平均距离,mindist的每个下标对应的就是act_layers的下标

    return avglist,test_act_layers  # 返回存储每个测试样本与每个集群中所有已选择测试样本之间的平均距离，格式为{0:[12,34,55......(数组长度均为所有测试样本总个数)],1:[32,45,34......],2:[34,56,67......]},

# 提取与所有集群中的距离特征，构建训练样本对和测试样本
def distance_feature(avglist, test_act_layers,flag):
    feature = [[] for i in range(len(test_act_layers))]
    for i in range(len(test_act_layers)):
        for key in avglist:
            if len(avglist[key]) == 0:
                continue
            feature[i].append(avglist[key][i])

    minMax = MinMaxScaler()
    feature = minMax.fit_transform(feature)
    feature = np.array(feature)

    feature_list = []
    if flag == 1:
        doc2 = [0] * len(avglist)
        for i in range(len(feature)):
            doc_pair = feature[i].tolist() + doc2
            feature_list.append(doc_pair)

        print("################test_feature#########")
        # print(feature_list[0])
        return feature_list

    n = 0
    for i in range(len(feature)):
        for j in range(len(feature)):
            if i == j:
                continue
            doc_pair = feature[i].tolist() + feature[j].tolist()
            feature_list.append(doc_pair)
            n = n + 1
    print("######################valid_feature######################")
    # print(n)
    # print(len(feature_list[0]))
    return feature_list

def new_select_from_mmd_Distance_valid(model,  x_valid, x_train,flag):
    res, train_act_layers, countlist = class_cluster_kmeans(model, x_train)
    selectlist = mmd_critic_centroid(res,train_act_layers)
    avglist,valid_act_layers = maxdistance_new(selectlist, train_act_layers,model,x_valid)
    valid_feature = distance_feature(avglist, valid_act_layers,flag)
    return valid_feature,res,train_act_layers

def new_select_from_mmd_Distance_test(res, train_act_layers, model,  x_test,flag):
    # res, train_act_layers, countlist = class_cluster_kmeans(model, x_train)
    selectlist = mmd_critic_centroid(res,train_act_layers)
    avglist,test_act_layers = maxdistance_new(selectlist, train_act_layers,model,x_test)
    test_feature = distance_feature(avglist, test_act_layers,flag)
    return test_feature

def learning_to_rank_val(val_feature, val_noise):

    # data preparation
    val_noise = np.array(val_noise)
    val_feature = np.array(val_feature)

    train_data = val_feature

    train_y = val_noise

    train_x = scipy.sparse.csr_matrix(train_data)
    # group_train = np.array([10000, 1000])

    train_dmatrix = DMatrix(train_x, train_y)
    params = {
        "objective": "rank:pairwise",
        "eta": 0.05,  #[0.01,0.05,0.1,0.15,0.2]  ,设置的默认值：0.05
        "eval_metric": "ndcg",
        "min_child_weight": 0.5,  #[0.1,0.3,0.5,0.7,0.9] 设置的默认值：0.5
        "max_depth": 10, #[2,4,6,8,10],  # 10 cifar10,mnist 设置的默认值：10
        "colsample_bytree":1
    }
    ranknet_model = xgb.train(params, train_dmatrix, evals=[(train_dmatrix, "train")])
    return ranknet_model

def learning_to_rank_test(dataset, ranknet_model,selectsize, x_test, y_test, test_feature):

    # data preparation
    test_feature = np.array(test_feature)

    test_data = test_feature
    test_x = scipy.sparse.csr_matrix(test_data)
    test_dmatrix = DMatrix(test_x)

    # L2R Algorithm
    ranknet_model_score = ranknet_model.predict(test_dmatrix)

    print("-----------ranknet_model_score--------------")
    # print(ranknet_model_score.tolist())
    rank_score = np.argsort(-ranknet_model_score)  # argsort函数返回数组值从小到大的索引值
    print("----------rank_score---------------")
    # print(rank_score)

    if dataset == 'mnist' or dataset == 'fashion_mnist':
        x = np.zeros((selectsize, 28, 28, 1))
        y = np.zeros((selectsize,))
    elif dataset == 'cifar10' or dataset == 'svhn':
        x = np.zeros((selectsize, 32, 32, 3))
        y = np.zeros((selectsize,))

    for i in range(selectsize):  # 根据排序，将最高分的测试样本筛选出来
        x[i] = x_test[rank_score[i]]
        y[i] = y_test[rank_score[i]]

    return x, y, rank_score




