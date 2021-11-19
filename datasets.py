import os
import numpy as np
import tensorflow as tf
from numpy import *
import matplotlib.pyplot as plt


# 读取图片
def get_file(path):
    bridge, label_bridge = [], []
    childs, label_childs = [], []
    downwarddog, label_downwarddog = [], []
    mountain, label_mountain = [], []
    plank, label_plank = [], []
    seatedforwardbend, label_seatedforwardbend = [], []
    tree, label_tree = [], []
    trianglepose, label_trianglepose = [], []
    warrior1, label_warrior1 = [], []
    warrior2, label_warrior2 = [], []
    # 获取数据
    for file in os.listdir(path+'/bridge'):
        bridge.append(path+'/bridge/'+file)
        label_bridge.append(1)
    for file in os.listdir(path+'/childs'):
        childs.append(path+'/childs/'+file)
        label_childs.append(2)
    for file in os.listdir(path+'/downwarddog'):
        downwarddog.append(path+'/downwarddog/'+file)
        label_downwarddog.append(3)
    for file in os.listdir(path+'/mountain'):
        mountain.append(path+'/mountain/'+file)
        label_mountain.append(4)
    for file in os.listdir(path+'/plank'):
        plank.append(path+'/plank/'+file)
        label_plank.append(5)
    for file in os.listdir(path+'/seatedforwardbend'):
        seatedforwardbend.append(path+'/seatedforwardbend/'+file)
        label_seatedforwardbend.append(6)
    for file in os.listdir(path+'/tree'):
        tree.append(path+'/tree/'+file)
        label_tree.append(7)
    for file in os.listdir(path+'/trianglepose'):
        trianglepose.append(path+'/trianglepose/'+file)
        label_trianglepose.append(8)
    for file in os.listdir(path+'/warrior1'):
        warrior1.append(path+'/warrior1/'+file)
        label_warrior1.append(9)
    for file in os.listdir(path+'/warrior2'):
        warrior2.append(path+'/warrior2/'+file)
        label_warrior2.append(10)
    # # 合并、转置、打乱
    # image_list = np.hstack((bridge, childs, downwarddog, mountain, plank, seatedforwardbend, tree, trianglepose, warrior1, warrior2))
    # label_list = np.hstack((label_bridge, label_childs, label_downwarddog, label_mountain, label_plank,
    #                        label_seatedforwardbend, label_tree, label_trianglepose, label_warrior1, label_warrior2))
    # 合并、转置、打乱
    image_list = np.hstack((bridge, childs, downwarddog, mountain))
    label_list = np.hstack((label_bridge, label_childs, label_downwarddog, label_mountain))
    # 利用shuffle，转置、随机打乱
    temp = np.array([image_list, label_list])
    temp = temp.transpose()
    # 打乱
    np.random.shuffle(temp)
    # 将所有的img和lab转换成list
    image_list = list(temp[:, 0])    # 图片路径
    label_list = list(temp[:, 1])    # 图片标签
    for i in range(len(label_list)):
        label_list[i] = int(label_list[i])
    # 返回图片list及其对应标签list
    return image_list, label_list


def preprocess(x, y):
    x = tf.cast(x, dtype=tf.float32) / 255
    y = tf.cast(y, dtype=tf.int32)
    y = tf.one_hot(y, depth=4)
    return x, y


def get_parameter():
    batch_size = 16
    # 训练数据集
    il, ll = get_file('dataset/train')
    x, y = [], []
    for i in range(len(il)):
        image = tf.io.read_file(il[i])
        image = tf.image.decode_jpeg(image, channels=3)
        x.append(image)
        y.append(ll[i])
    x = np.array(x)
    y = np.array(y)
    ds = tf.data.Dataset.from_tensor_slices((x, y))
    ds = ds.map(preprocess).batch(batch_size)
    # 测试数据集
    il_val, ll_val = get_file('dataset/test')
    x_val, y_val = [], []
    for j in range(len(il_val)):
        image_val = tf.io.read_file(il_val[j])
        image_val = tf.image.decode_jpeg(image_val, channels=3)
        x_val.append(image_val)
        y_val.append(ll_val[j])
    x_val = np.array(x_val)
    y_val = np.array(y_val)
    ds_val = tf.data.Dataset.from_tensor_slices((x_val, y_val))
    ds_val = ds_val.map(preprocess).batch(batch_size)

    return ds, ds_val


if __name__ == '__main__':
    get_parameter()