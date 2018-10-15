"""Test ImageNet pretrained DenseNet"""
from __future__ import print_function
import sys
# sys.path.insert(0,'/home/xmli/livertumor_xmli/Keras-2.0.8')
# sys.path.insert(0,'/home/xmli/livertumor_xmli/mylib')
# sys.path.insert(0,'/research/pheng/xmli/livertumor/Keras-2.0.8')
# sys.path.insert(0,'/research/pheng/xmli/livertumor/mylib')
sys.path.insert(0,'/home/xmli/big/livertumor/Keras-2.0.8')
sys.path.insert(0,'/home/xmli/big/livertumor/mylib')
from multiprocessing.dummy import Pool as ThreadPool
import random
import glob
from keras.utils.vis_utils import plot_model
from medpy.io import load
import numpy as np
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint
import tensorflow as tf
from keras.models import Model
from keras.layers import Input, ZeroPadding2D, concatenate, Lambda, Reshape, LSTM, ConvLSTM2D, Permute, ZeroPadding3D, add
from keras.layers.core import Dropout, Activation
from keras.layers.convolutional import UpSampling2D, Conv2D, Conv3D, UpSampling3D, AveragePooling3D
from keras.layers.pooling import AveragePooling2D, MaxPooling2D, MaxPooling3D
from keras.layers.normalization import BatchNormalization
import keras.backend as K
import os
import time
from skimage.transform import resize
# from keras.utils2.multi_gpu import make_parallel
from custom_layers import Scale
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
K.set_image_dim_ordering('tf')

path = './result_train_dense167_fast_bauto3d_residualconv_endtoend_big/'
batch_size = 1
img_deps = 224
img_rows = 224
img_cols = 8
std = 37
thread_num = 14
txtfile = 'myTrainingDataTxt'
mean = 48

liverlist = [32,34,38,41,47,87,89,91,105,106,114,115,119]
DataList = ["/home/xmli/gpu7_xmli/"]
# DataList = ["/data/ssd/public/xmli/Data_gpu7/"]
def load_seq_crop_data_masktumor_try(Parameter_List):
    img = Parameter_List[0]
    tumor = Parameter_List[1]
    lines = Parameter_List[2]
    numid = Parameter_List[3]
    minindex = Parameter_List[4]
    maxindex = Parameter_List[5]
    #  randomly scale
    scale = np.random.uniform(0.8,1.2)
    deps = int(img_deps * scale)
    rows = int(img_rows * scale)
    cols = img_cols

    sed = np.random.randint(1,numid)
    cen = lines[sed-1]
    cen = np.fromstring(cen, dtype=int, sep=' ')
    # print (cen)
    a = min(max(minindex[0] + deps/2, cen[0]), maxindex[0]- deps/2-1)
    b = min(max(minindex[1] + rows/2, cen[1]), maxindex[1]- rows/2-1)
    c = min(max(minindex[2] + cols/2, cen[2]), maxindex[2]- cols/2-1)
    cropp_img = img[a - deps / 2:a + deps / 2, b - rows / 2:b + rows / 2,
                c - img_cols / 2: c + img_cols / 2].copy()
    cropp_tumor = tumor[a - deps / 2:a + deps / 2, b - rows / 2:b + rows / 2,
                  c - img_cols / 2:c + img_cols / 2].copy()

    cropp_img -= mean
     # randomly flipping
    flip_num = np.random.randint(0,8)
    if flip_num == 1:
        cropp_img = np.flipud(cropp_img)
        cropp_tumor = np.flipud(cropp_tumor)
    elif flip_num == 2:
        cropp_img = np.fliplr(cropp_img)
        cropp_tumor = np.fliplr(cropp_tumor)
    elif flip_num == 3:
        cropp_img = np.rot90(cropp_img, k=1, axes=(1, 0))
        cropp_tumor = np.rot90(cropp_tumor, k=1, axes=(1, 0))
    elif flip_num == 4:
        cropp_img = np.rot90(cropp_img, k=3, axes=(1, 0))
        cropp_tumor = np.rot90(cropp_tumor, k=3, axes=(1, 0))
    elif flip_num == 5:
        cropp_img = np.fliplr(cropp_img)
        cropp_tumor = np.fliplr(cropp_tumor)
        cropp_img = np.rot90(cropp_img, k=1, axes=(1, 0))
        cropp_tumor = np.rot90(cropp_tumor, k=1, axes=(1, 0))
    elif flip_num == 6:
        cropp_img = np.fliplr(cropp_img)
        cropp_tumor = np.fliplr(cropp_tumor)
        cropp_img = np.rot90(cropp_img, k=3, axes=(1, 0))
        cropp_tumor = np.rot90(cropp_tumor, k=3, axes=(1, 0))
    elif flip_num == 7:
        cropp_img = np.flipud(cropp_img)
        cropp_tumor = np.flipud(cropp_tumor)
        cropp_img = np.fliplr(cropp_img)
        cropp_tumor = np.fliplr(cropp_tumor)
    #
    cropp_tumor = resize(cropp_tumor, (img_deps,img_rows,img_cols), order=0, mode='edge', cval=0, clip=True, preserve_range=True)
    cropp_img   = resize(cropp_img, (img_deps,img_rows,img_cols), order=3, mode='constant', cval=0, clip=True, preserve_range=True)
    return cropp_img, cropp_tumor
def generate_arrays_from_file(batch_size, trainidx, img_list, tumor_list, tumorlines, liverlines, tumoridx, liveridx, minindex_list, maxindex_list):
    while 1:
        X = np.zeros((batch_size, img_deps, img_rows, img_cols,1), dtype='float32')
        Y = np.zeros((batch_size, img_deps, img_rows, img_cols,1), dtype='int16')
        Parameter_List = []
        for idx in xrange(batch_size):
            count = np.random.choice(trainidx)
            img = img_list[count]
            tumor = tumor_list[count]
            minindex = minindex_list[count]
            maxindex = maxindex_list[count]
            num = np.random.randint(0,6)
            if num < 3 or (count in liverlist):
                lines = liverlines[count]
                numid = liveridx[count]
            else:
                lines = tumorlines[count]
                numid = tumoridx[count]
            Parameter_List.append([img, tumor, lines, numid, minindex, maxindex])
        pool = ThreadPool(thread_num)
        result_list = pool.map(load_seq_crop_data_masktumor_try, Parameter_List)
        pool.close()
        pool.join()
        for idx in xrange(len(result_list)):
            X[idx, :, :, :, 0] = result_list[idx][0]
            Y[idx, :, :, :, 0] = result_list[idx][1]
        if np.sum(Y==0)==0:
            continue
        if np.sum(Y==1)==0:
            continue
        if np.sum(Y==2)==0:
            continue
        yield (X,Y)

def weighted_crossentropy(y_true, y_pred):
    y_pred = y_pred[:,:,:,1:7,:]
    y_true = y_true[:,:,:,1:7,:]
    y_pred_f = K.reshape(y_pred, (batch_size*img_deps*img_rows*6,3))
    y_true_f = K.reshape(y_true, (batch_size*img_deps*img_rows*6,))

    soft_pred_f = K.softmax(y_pred_f)
    soft_pred_f = K.log(tf.clip_by_value(soft_pred_f, 1e-10, 1.0))

    neg = K.equal(y_true_f, K.zeros_like(y_true_f))
    neg_calculoss = tf.gather(soft_pred_f[:,0], tf.where(neg))

    pos1 = K.equal(y_true_f, K.ones_like(y_true_f))
    pos1_calculoss = tf.gather(soft_pred_f[:,1], tf.where(pos1))

    pos2 = K.equal(y_true_f, 2*K.ones_like(y_true_f))
    pos2_calculoss = tf.gather(soft_pred_f[:,2], tf.where(pos2))

    loss = -K.mean(tf.concat([0.78*neg_calculoss, 0.65*pos1_calculoss, 8.57*pos2_calculoss], 0))

    # # hard mining
    # num = img_deps*img_rows*batch_size*6/20.
    # # values, index = tf.nn.top_k(-calculoss, k=num)
    # # newcalculoss = tf.gather(calculoss, index)
    # # loss = -K.mean(newcalculoss)
    # #
    # ind2 = K.sum(tf.cast(pos2, tf.float32))
    # ind1 = K.sum(tf.cast(pos1, tf.float32))
    # ind0 = K.sum(tf.cast(neg, tf.float32))
    # ind = K.minimum(ind2,ind1)
    # ind = K.minimum(ind0,ind)
    #
    # num = K.minimum(ind,num)
    # num = tf.to_int32(num)
    #
    # # #  hard mining respectively
    #
    # neg  = tf.reshape(tf.cast(neg,  tf.int32), [-1,])
    # pos1 = tf.reshape(tf.cast(pos1, tf.int32), [-1,])
    # pos2 = tf.reshape(tf.cast(pos2, tf.int32), [-1,])
    #
    # select_neg = tf.dynamic_partition(soft_pred_f[:,0], neg, 2)
    # select_neg = select_neg[1]
    # val_neg, idx_neg = tf.nn.top_k(-select_neg, k= num)
    #
    # select_pos1 = tf.dynamic_partition(soft_pred_f[:,1], pos1, 2)
    # select_pos1 = select_pos1[1]
    # val_pos1, idx_pos1 = tf.nn.top_k(-select_pos1, k= num)
    #
    # select_pos2 = tf.dynamic_partition(soft_pred_f[:,2], pos2, 2)
    # select_pos2 = select_pos2[1]
    # val_pos2, idx_pos2 = tf.nn.top_k(-select_pos2, k=num)
    #
    # loss = K.mean(tf.concat([val_neg, val_pos1, val_pos2], 0))


    return loss
def conv_block3d(x, stage, branch, nb_filter, dropout_rate=None, weight_decay=1e-4):
    '''Apply BatchNorm, Relu, bottleneck 1x1 Conv3D, 3x3 Conv3D, and option dropout
        # Arguments
            x: input tensor 
            stage: index for dense block
            branch: layer index within each dense block
            nb_filter: number of filters
            dropout_rate: dropout rate
            weight_decay: weight decay factor
    '''
    eps = 1.1e-5
    conv_name_base = '3dconv' + str(stage) + '_' + str(branch)
    relu_name_base = '3drelu' + str(stage) + '_' + str(branch)

    # 1x1 Convolution (Bottleneck layer)
    inter_channel = nb_filter * 4
    x = BatchNormalization(epsilon=eps, axis=4, name=conv_name_base+'_x1_bn', momentum=1.0, trainable=False)(x, training=False)
    x = Scale(axis=4, name=conv_name_base+'_x1_scale')(x)
    x = Activation('relu', name=relu_name_base+'_x1')(x)
    x = Conv3D(inter_channel, (1, 1, 1), name=conv_name_base+'_x1', use_bias=False)(x)

    if dropout_rate:
        x = Dropout(dropout_rate)(x)

    # 3x3 Convolution
    x = BatchNormalization(epsilon=eps, axis=4, name=conv_name_base+'_x2_bn', momentum=1.0, trainable=False)(x, training=False)
    x = Scale(axis=4, name=conv_name_base+'_x2_scale')(x)
    x = Activation('relu', name=relu_name_base+'_x2')(x)
    x = ZeroPadding3D((1, 1, 1), name=conv_name_base+'_x2_zeropadding')(x)
    x = Conv3D(nb_filter, (3, 3, 3), name=conv_name_base+'_x2', use_bias=False)(x)

    if dropout_rate:
        x = Dropout(dropout_rate)(x)

    return x
def dense_block3d(x, stage, nb_layers, nb_filter, growth_rate, dropout_rate=None, weight_decay=1e-4, grow_nb_filters=True):
    ''' Build a dense_block where the output of each conv_block is fed to subsequent ones
        # Arguments
            x: input tensor
            stage: index for dense block
            nb_layers: the number of layers of conv_block to append to the model.
            nb_filter: number of filters
            growth_rate: growth rate
            dropout_rate: dropout rate
            weight_decay: weight decay factor
            grow_nb_filters: flag to decide to allow number of filters to grow
    '''

    eps = 1.1e-5
    concat_feat = x

    for i in range(nb_layers):
        branch = i+1
        x = conv_block3d(concat_feat, stage, branch, growth_rate, dropout_rate, weight_decay)
        concat_feat = concatenate([concat_feat, x], axis=4, name='3dconcat_'+str(stage)+'_'+str(branch))

        if grow_nb_filters:
            nb_filter += growth_rate

    return concat_feat, nb_filter
def transition_block3d(x, stage, nb_filter, compression=1.0, dropout_rate=None, weight_decay=1E-4):
    ''' Apply BatchNorm, 1x1 Convolution, averagePooling, optional compression, dropout 
        # Arguments
            x: input tensor
            stage: index for dense block
            nb_filter: number of filters
            compression: calculated as 1 - reduction. Reduces the number of feature maps in the transition block.
            dropout_rate: dropout rate
            weight_decay: weight decay factor
    '''

    eps = 1.1e-5
    conv_name_base = '3dconv' + str(stage) + '_blk'
    relu_name_base = '3drelu' + str(stage) + '_blk'
    pool_name_base = '3dpool' + str(stage)

    x = BatchNormalization(epsilon=eps, axis=4, name=conv_name_base+'_bn', momentum=1.0)(x, training=False)
    x = Scale(axis=4, name=conv_name_base+'_scale')(x)
    x = Activation('relu', name=relu_name_base)(x)
    x = Conv3D(int(nb_filter * compression), (1, 1, 1), name=conv_name_base, use_bias=False)(x)

    if dropout_rate:
        x = Dropout(dropout_rate)(x)

    x = AveragePooling3D((2, 2, 1), strides=(2, 2, 1), name=pool_name_base)(x)

    return x
def DenseNet3D(img_input, nb_dense_block=4, growth_rate=32, nb_filter=96, reduction=0.0, dropout_rate=0.0, weight_decay=1e-4, classes=1000, weights_path=None):
    '''Instantiate the DenseNet 161 architecture,
        # Arguments
            nb_dense_block: number of dense blocks to add to end
            growth_rate: number of filters to add per dense block
            nb_filter: initial number of filters
            reduction: reduction factor of transition blocks.
            dropout_rate: dropout rate
            weight_decay: weight decay factor
            classes: optional number of classes to classify images
            weights_path: path to pre-trained weights
        # Returns
            A Keras model instance.
    '''
    eps = 1.1e-5

    # compute compression factor
    compression = 1.0 - reduction

    # From architecture for ImageNet (Table 1 in the paper)
    nb_filter = 96
    nb_layers = [3, 4, 12, 8]  # For DenseNet-161
    box = []
    # Initial convolution
    x = ZeroPadding3D((3, 3, 3), name='3dconv1_zeropadding')(img_input)
    x = Conv3D(nb_filter, (7, 7, 7), strides=(2, 2, 2), name='3dconv1', use_bias=False)(x)
    x = BatchNormalization(epsilon=eps, axis=4, name='3dconv1_bn')(x)
    x = Scale(axis=4, name='3dconv1_scale')(x)
    x = Activation('relu', name='3drelu1')(x)
    box.append(x)
    x = ZeroPadding3D((1, 1, 1), name='3dpool1_zeropadding')(x)
    x = MaxPooling3D((3, 3, 3), strides=(2, 2, 2), name='3dpool1')(x)

    # Add dense blocks
    for block_idx in range(nb_dense_block - 1):
        stage = block_idx + 2
        x, nb_filter = dense_block3d(x, stage, nb_layers[block_idx], nb_filter, growth_rate, dropout_rate=dropout_rate,
                                   weight_decay=weight_decay)
        box.append(x)
        # Add transition_block
        x = transition_block3d(x, stage, nb_filter, compression=compression, dropout_rate=dropout_rate,
                             weight_decay=weight_decay)
        nb_filter = int(nb_filter * compression)

    final_stage = stage + 1
    x, nb_filter = dense_block3d(x, final_stage, nb_layers[-1], nb_filter, growth_rate, dropout_rate=dropout_rate,
                               weight_decay=weight_decay)

    x = BatchNormalization(epsilon=eps, axis=4, name='3dconv' + str(final_stage) + '_blk_bn')(x)
    x = Scale(axis=4, name='3dconv' + str(final_stage) + '_blk_scale')(x)
    x = Activation('relu', name='3drelu' + str(final_stage) + '_blk')(x)
    box.append(x)
    print (box)
    up0 = UpSampling3D(size=(2, 2, 1))(x)
    # line0 = Conv3D(504, (1, 1, 1), padding="same", name="3dline0")(box[3])
    # up0_sum = add([line0, up0])
    conv_up0 = Conv3D(504, (3, 3, 3), padding="same", name="3dconv_up0")(up0)
    bn_up0 = BatchNormalization(name="3dbn_up0")(conv_up0)
    ac_up0 = Activation('relu', name='3dac_up0')(bn_up0)

    up1 = UpSampling3D(size=(2, 2, 1))(ac_up0)
    # up1_sum = add([box[2], up1])
    conv_up1 = Conv3D(224, (3, 3, 3), padding="same", name="3dconv_up1")(up1)
    bn_up1 = BatchNormalization(name="3dbn_up1")(conv_up1)
    ac_up1 = Activation('relu', name='3dac_up1')(bn_up1)

    up2 = UpSampling3D(size=(2, 2, 1))(ac_up1)
    # up2_sum = add([box[1], up2])
    conv_up2 = Conv3D(192, (3, 3, 3), padding="same", name="3dconv_up2")(up2)
    bn_up2 = BatchNormalization(name="3dbn_up2")(conv_up2)
    ac_up2 = Activation('relu', name='3dac_up2')(bn_up2)

    up3 = UpSampling3D(size=(2, 2, 2))(ac_up2)
    # up3_sum = add([box[0], up3])
    conv_up3 = Conv3D(96, (3, 3, 3), padding="same", name="3dconv_up3")(up3)
    bn_up3 = BatchNormalization(name="3dbn_up3")(conv_up3)
    ac_up3 = Activation('relu', name='3dac_up3')(bn_up3)

    up4 = UpSampling3D(size=(2, 2, 2))(ac_up3)
    conv_up4 = Conv3D(64, (3, 3, 3), padding="same", name="3dconv_up4")(up4)
    bn_up4 = BatchNormalization(name="3dbn_up4")(conv_up4)
    ac_up4 = Activation('relu', name='3dac_up4')(bn_up4)

    x = Conv3D(3, (1, 1, 1), padding="same", name='3dclassifer')(ac_up4)

    return ac_up4, x



def DenseUNet(img_input, nb_dense_block=4, growth_rate=48, nb_filter=96, reduction=0.0, dropout_rate=0.0, weight_decay=1e-4, classes=1000, weights_path=None):
    '''Instantiate the DenseNet 161 architecture,
        # Arguments
            nb_dense_block: number of dense blocks to add to end
            growth_rate: number of filters to add per dense block
            nb_filter: initial number of filters
            reduction: reduction factor of transition blocks.
            dropout_rate: dropout rate
            weight_decay: weight decay factor
            classes: optional number of classes to classify images
            weights_path: path to pre-trained weights
        # Returns
            A Keras model instance.
    '''
    eps = 1.1e-5
    # compute compression factor
    compression = 1.0 - reduction

    # Handle Dimension Ordering for different backends
    global concat_axis
    concat_axis = 3

    # From architecture for ImageNet (Table 1 in the paper)
    nb_filter = 96
    nb_layers = [6,12,36,24] # For DenseNet-161
    box = []
    # Initial convolution
    x = ZeroPadding2D((3, 3), name='conv1_zeropadding')(img_input)
    x = Conv2D(nb_filter, (7, 7), strides=(2, 2), name='conv1', use_bias=False, trainable=True)(x)
    x = BatchNormalization(epsilon=eps, axis=concat_axis, momentum = 1, name='conv1_bn', trainable=False)(x, training=False)
    x = Scale(axis=concat_axis, name='conv1_scale')(x)
    x = Activation('relu', name='relu1')(x)
    box.append(x)
    x = ZeroPadding2D((1, 1), name='pool1_zeropadding')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), name='pool1')(x)

    # Add dense blocks
    for block_idx in range(nb_dense_block - 1):
        stage = block_idx+2
        x, nb_filter = dense_block(x, stage, nb_layers[block_idx], nb_filter, growth_rate, dropout_rate=dropout_rate, weight_decay=weight_decay)
        box.append(x)
        # Add transition_block
        x = transition_block(x, stage, nb_filter, compression=compression, dropout_rate=dropout_rate, weight_decay=weight_decay)
        nb_filter = int(nb_filter * compression)

    final_stage = stage + 1
    x, nb_filter = dense_block(x, final_stage, nb_layers[-1], nb_filter, growth_rate, dropout_rate=dropout_rate, weight_decay=weight_decay)

    x = BatchNormalization(epsilon=eps, axis=concat_axis, momentum = 1, name='conv'+str(final_stage)+'_blk_bn', trainable=False)(x, training=False)
    x = Scale(axis=concat_axis, name='conv'+str(final_stage)+'_blk_scale')(x)
    x = Activation('relu', name='relu'+str(final_stage)+'_blk')(x)
    box.append(x)

    up0 = UpSampling2D(size=(2,2))(x)
    conv_up0 = Conv2D(768, (3, 3), padding="same", name = "conv_up0", trainable=True)(up0)
    bn_up0 = BatchNormalization(name = "bn_up0", momentum = 1, trainable=False)(conv_up0, training=False)
    ac_up0 = Activation('relu', name='ac_up0')(bn_up0)

    up1 = UpSampling2D(size=(2,2))(ac_up0)
    conv_up1 = Conv2D(384, (3, 3), padding="same", name = "conv_up1", trainable=True)(up1)
    bn_up1 = BatchNormalization(name = "bn_up1", momentum = 1, trainable=False)(conv_up1, training=False)
    ac_up1 = Activation('relu', name='ac_up1')(bn_up1)

    up2 = UpSampling2D(size=(2,2))(ac_up1)
    conv_up2 = Conv2D(96, (3, 3), padding="same", name = "conv_up2", trainable=True)(up2)
    bn_up2 = BatchNormalization(name = "bn_up2", momentum = 1, trainable=False)(conv_up2, training=False)
    ac_up2 = Activation('relu', name='ac_up2')(bn_up2)

    up3 = UpSampling2D(size=(2,2))(ac_up2)
    conv_up3 = Conv2D(96, (3, 3), padding="same", name = "conv_up3", trainable=True)(up3)
    bn_up3 = BatchNormalization(name = "bn_up3", momentum = 1, trainable=False)(conv_up3, training=False)
    ac_up3 = Activation('relu', name='ac_up3')(bn_up3)

    up4 = UpSampling2D(size=(2, 2))(ac_up3)
    conv_up4 = Conv2D(64, (3, 3), padding="same", name="conv_up4", trainable=True)(up4)
    bn_up4 = BatchNormalization(name="bn_up4", momentum = 1, trainable=False)(conv_up4, training=False)
    ac_up4 = Activation('relu', name='ac_up4')(bn_up4)

    x = Conv2D(3, (1,1), padding="same", name='dense167classifer', trainable=True)(ac_up4)

    return ac_up4, x

def conv_block(x, stage, branch, nb_filter, dropout_rate=None, weight_decay=1e-4):
    '''Apply BatchNorm, Relu, bottleneck 1x1 Conv2D, 3x3 Conv2D, and option dropout
        # Arguments
            x: input tensor 
            stage: index for dense block
            branch: layer index within each dense block
            nb_filter: number of filters
            dropout_rate: dropout rate
            weight_decay: weight decay factor
    '''
    eps = 1.1e-5
    conv_name_base = 'conv' + str(stage) + '_' + str(branch)
    relu_name_base = 'relu' + str(stage) + '_' + str(branch)

    # 1x1 Convolution (Bottleneck layer)
    inter_channel = nb_filter * 4
    x = BatchNormalization(epsilon=eps, axis=concat_axis, momentum = 1, name=conv_name_base+'_x1_bn', trainable=False)(x, training=False)
    x = Scale(axis=concat_axis, name=conv_name_base+'_x1_scale')(x)
    x = Activation('relu', name=relu_name_base+'_x1')(x)
    x = Conv2D(inter_channel, (1, 1), name=conv_name_base+'_x1', use_bias=False, trainable=True)(x)

    if dropout_rate:
        x = Dropout(dropout_rate)(x)

    # 3x3 Convolution
    x = BatchNormalization(epsilon=eps, axis=concat_axis, momentum = 1, name=conv_name_base+'_x2_bn', trainable=False)(x, training=False)
    x = Scale(axis=concat_axis, name=conv_name_base+'_x2_scale')(x)
    x = Activation('relu', name=relu_name_base+'_x2')(x)
    x = ZeroPadding2D((1, 1), name=conv_name_base+'_x2_zeropadding')(x)
    x = Conv2D(nb_filter, (3, 3), name=conv_name_base+'_x2', use_bias=False, trainable=True)(x)

    if dropout_rate:
        x = Dropout(dropout_rate)(x)

    return x


def transition_block(x, stage, nb_filter, compression=1.0, dropout_rate=None, weight_decay=1E-4):
    ''' Apply BatchNorm, 1x1 Convolution, averagePooling, optional compression, dropout 
        # Arguments
            x: input tensor
            stage: index for dense block
            nb_filter: number of filters
            compression: calculated as 1 - reduction. Reduces the number of feature maps in the transition block.
            dropout_rate: dropout rate
            weight_decay: weight decay factor
    '''

    eps = 1.1e-5
    conv_name_base = 'conv' + str(stage) + '_blk'
    relu_name_base = 'relu' + str(stage) + '_blk'
    pool_name_base = 'pool' + str(stage)

    x = BatchNormalization(epsilon=eps, axis=concat_axis, momentum = 1, name=conv_name_base+'_bn', trainable=False)(x, training=False)
    x = Scale(axis=concat_axis, name=conv_name_base+'_scale')(x)
    x = Activation('relu', name=relu_name_base)(x)
    x = Conv2D(int(nb_filter * compression), (1, 1), name=conv_name_base, use_bias=False, trainable=True)(x)

    if dropout_rate:
        x = Dropout(dropout_rate)(x)

    x = AveragePooling2D((2, 2), strides=(2, 2), name=pool_name_base)(x)

    return x


def dense_block(x, stage, nb_layers, nb_filter, growth_rate, dropout_rate=None, weight_decay=1e-4, grow_nb_filters=True):
    ''' Build a dense_block where the output of each conv_block is fed to subsequent ones
        # Arguments
            x: input tensor
            stage: index for dense block
            nb_layers: the number of layers of conv_block to append to the model.
            nb_filter: number of filters
            growth_rate: growth rate
            dropout_rate: dropout rate
            weight_decay: weight decay factor
            grow_nb_filters: flag to decide to allow number of filters to grow
    '''

    eps = 1.1e-5
    concat_feat = x

    for i in range(nb_layers):
        branch = i+1
        x = conv_block(concat_feat, stage, branch, growth_rate, dropout_rate, weight_decay)
        concat_feat = concatenate([concat_feat, x], axis=concat_axis, name='concat_'+str(stage)+'_'+str(branch))

        if grow_nb_filters:
            nb_filter += growth_rate

    return concat_feat, nb_filter
def slice(x, h1, h2):
    """ Define a tensor slice function 
    """
    return x[:, :, :, h1:h2,:]
def slice2d(x, h1, h2):

    tmp = x[h1:h2,:,:,:]
    tmp = tf.transpose(tmp, perm=[1, 2, 0, 3])
    tmp = tf.expand_dims(tmp, 0)
    return tmp

def slice_last(x):

    x = x[:,:,:,:,0]
    return x
def trans(x):

    x = tf.transpose(x, perm=[0,3,1,2,4])
    return x
def trans_back(x):

    x = tf.transpose(x, perm=[0,2,3,1,4])

    return x
def dense_rnn_net():

    #  ************************3d volume input******************************************************************
    img_input = Input(batch_shape=(batch_size, img_deps, img_rows, img_cols, 1), name='volumetric_data')

    #  ************************(batch*d3cols)*2dvolume--2D DenseNet branch**************************************
    input2d = Lambda(slice, arguments={'h1': 0, 'h2': 2})(img_input)
    single = Lambda(slice, arguments={'h1':0, 'h2':1})(img_input)
    input2d = concatenate([single, input2d], axis=3)
    for i in xrange(img_cols - 2):
        input2d_tmp = Lambda(slice, arguments={'h1': i, 'h2': i + 3})(img_input)
        input2d = concatenate([input2d, input2d_tmp], axis=0)
        if i == img_cols - 3:
            final1 = Lambda(slice, arguments={'h1': img_cols-2, 'h2': img_cols})(img_input)
            final2 = Lambda(slice, arguments={'h1': img_cols-1, 'h2': img_cols})(img_input)
            final = concatenate([final1, final2], axis=3)
            input2d = concatenate([input2d, final], axis=0)
    input2d = Lambda(slice_last)(input2d)

    #  ******************************stack to 3D volumes *******************************************************
    feature2d, classifer2d = DenseUNet(input2d, reduction=0.5)
    res2d = Lambda(slice2d, arguments={'h1': 0, 'h2': 1})(classifer2d)
    fea2d = Lambda(slice2d, arguments={'h1':0, 'h2':1})(feature2d)
    for j in xrange(img_cols - 1):
        score = Lambda(slice2d, arguments={'h1': j + 1, 'h2': j + 2})(classifer2d)
        fea2d_slice = Lambda(slice2d, arguments={'h1': j + 1, 'h2': j + 2})(feature2d)
        res2d = concatenate([res2d, score], axis=3)
        fea2d = concatenate([fea2d, fea2d_slice], axis=3)

    #  *************************** 3d DenseNet on 3D volume (concate with feature map )*********************************
    res2d_input = Lambda(lambda x: x * 250)(res2d)
    input3d_ori = Lambda(slice, arguments={'h1': 0, 'h2': img_cols})(img_input)
    input3d = concatenate([input3d_ori, res2d_input], axis=4)
    feature3d, classifer3d = DenseNet3D(input3d, reduction=0.5)

    final = add([feature3d, fea2d])
    final_conv = Conv3D(64, (3, 3, 3), padding="same", name='fianl_conv')(final)
    final_conv = Dropout(rate=0.3)(final_conv)
    final_bn = BatchNormalization(name="final_bn")(final_conv)
    final_ac = Activation('relu', name='final_ac')(final_bn)
    classifer = Conv3D(3, (1, 1, 1), padding="same", name='2d3dclassifer')(final_ac)

    print (classifer)
    model = Model( inputs = img_input,outputs = classifer, name='auto3d_residual_conv')

    return model

def train_and_predict():

    print('-'*30)
    print('Creating and compiling model...')
    print('-'*30)
    model = dense_rnn_net()
    sgd = SGD(lr=1e-3, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss=[weighted_crossentropy])
    model.load_weights(path+'/model/weights-drop3-3.46-0.02.hdf5')
    print (model.summary())

    # names = [weight.name for layer in model.layers for weight in layer.weights]
    # print (names)
    # print (len(names))
    #
    # weights = model.get_weights()
    # i = 0
    # for name, weight in zip(names, weights):
    #     # print (name, weight)
    #     i = i + 1
    #     if i > 1155:
    #         print (name, weight)
    #         # if (i==10):
    #         #     exit(0)


    #  liver tumor LITS
    trainidx = list(range(131))
    img_list = []
    tumor_list = []
    minindex_list = []
    maxindex_list = []
    tumorlines = []
    tumoridx = []
    liveridx = []
    liverlines = []
    t1=time.time()
    for idx in xrange(131):
        img, img_header = load(DataList[0] + 'myTrainingData/volume-' + str(idx) + '.nii' )
        tumor, tumor_header = load(DataList[0] + 'myTrainingData/segmentation-' + str(idx) + '.nii')
        img_list.append(img)
        tumor_list.append(tumor)

        maxmin = np.loadtxt(DataList[0] + str(txtfile) + '/LiverBox/box_' + str(idx) + '.txt', delimiter=' ')
        minindex = maxmin[0:3]
        maxindex = maxmin[3:6]
        minindex = np.array(minindex, dtype='int')
        maxindex = np.array(maxindex, dtype='int')
        minindex[0] = max(minindex[0]-3, 0)
        minindex[1] = max(minindex[1]-3, 0)
        minindex[2] = max(minindex[2]-3, 0)
        maxindex[0] = min(img.shape[0], maxindex[0]+3)
        maxindex[1] = min(img.shape[1], maxindex[1]+3)
        maxindex[2] = min(img.shape[2], maxindex[2]+3)
        minindex_list.append(minindex)
        maxindex_list.append(maxindex)

        f1 = open(DataList[0] + str(txtfile) + '/TumorPixels/tumor_' + str(idx) + '.txt','r')
        tumorline = f1.readlines()
        tumorlines.append(tumorline)
        tumoridx.append(len(tumorline))
        f1.close()

        f2 = open(DataList[0] + str(txtfile) + '/LiverPixels/liver_' + str(idx) + '.txt','r')
        liverline = f2.readlines()
        liverlines.append(liverline)
        liveridx.append(len(liverline))
        f2.close()
    t2=time.time()
    print (t2-t1)

    # c = [4, 2, 10, 8, 5, 3, 7, 14, 6, 13, 11,  9,  1, 12,  0]
    # fold_index1 = [4, 2, 10, 8, 5, 3, 7, 14, 6, 13]
    # fold_index2 = [3, 7, 14, 6, 13, 11,  9,  1, 12,  0]
    # fold_index3 = [4, 2, 10, 8, 5, 11,  9,  1, 12,  0]
    # fold_index = fold_index1
    # # if opt.name == '3Dircadb_1fold':
    # #     fold_index = fold_index1
    # # elif opt.name == '3Dircadb_2fold':
    # #     fold_index = fold_index2
    # # else:
    # #     fold_index = fold_index3
    # dataroot = '/home/xmli/gpu7_xmli/'
    # img_path = glob.glob(dataroot + '/3DIRCADb/3Dircadb1.*/myimage_*.nii')
    # label_path = glob.glob(dataroot + '/3DIRCADb/3Dircadb1.*/label.nii')
    # box_path = glob.glob(dataroot + '/3DIRCADb/3Dircadb1.*/box.txt')
    # tumortxt_path = glob.glob(dataroot + '/3DIRCADb/3Dircadb1.*/tumor.txt')
    # livertxt_path = glob.glob(dataroot + '/3DIRCADb/3Dircadb1.*/liver.txt')

    # print (box_path)
    # print (tumortxt_path)
    # print (livertxt_path)
    # # # exit(0)
    # trainidx = list(range(10))
    # img_list = []
    # tumor_list = []
    # minindex_list = []
    # maxindex_list = []
    # tumorlines = []
    # tumoridx = []
    # liveridx = []
    # liverlines = []
    # t1=time.time()
    # for idx in xrange(10):
    #     img, img_header = load(img_path[fold_index[idx]])
    #     tumor, tumor_header = load(label_path[fold_index[idx]])
    #     img_list.append(img)
    #     tumor_list.append(tumor)
    #
    #     #
    #     # print (img_path[fold_index[idx]], box_path[fold_index[idx]], tumortxt_path[fold_index[idx]], livertxt_path[fold_index[idx]])
    #     maxmin = np.loadtxt(box_path[idx], delimiter=' ')
    #     minindex = maxmin[0:3]
    #     maxindex = maxmin[3:6]
    #     minindex = np.array(minindex, dtype='int')
    #     maxindex = np.array(maxindex, dtype='int')
    #     minindex[0] = max(minindex[0]-3, 0)
    #     minindex[1] = max(minindex[1]-3, 0)
    #     minindex[2] = max(minindex[2]-3, 0)
    #     maxindex[0] = min(img.shape[0], maxindex[0]+3)
    #     maxindex[1] = min(img.shape[1], maxindex[1]+3)
    #     maxindex[2] = min(img.shape[2], maxindex[2]+3)
    #     minindex_list.append(minindex)
    #     maxindex_list.append(maxindex)
    #
    #     f1 = open(tumortxt_path[idx],'r')
    #     tumorline = f1.readlines()
    #     tumorlines.append(tumorline)
    #     tumoridx.append(len(tumorline))
    #     f1.close()
    #
    #     f2 = open(livertxt_path[idx],'r')
    #     liverline = f2.readlines()
    #     liverlines.append(liverline)
    #     liveridx.append(len(liverline))
    #     f2.close()
    #
    # print ('Number of traing images ' + str(len(img_list)))
    # print ('Number of traing labels ' + str(len(tumor_list)))


    if not os.path.exists(path + "model_u"):
        os.mkdir(path + 'model_u')
        os.mkdir(path + 'history')
    # else:
    #     if os.path.exists(path + "history/lossbatch.txt"):
    #         os.remove(path + 'history/lossbatch.txt')
    #     if os.path.exists(path + "history/lossepoch.txt"):
    #         os.remove(path + 'history/lossepoch.txt')
    model_checkpoint = ModelCheckpoint(path + 'model_u/weights.{epoch:02d}-{loss:.2f}.hdf5', monitor='loss', verbose = 1,
                                       save_best_only=False,save_weights_only=False,mode = 'min', period = 1)

    print('-'*30)
    print('Fitting model......')
    print('-'*30)

    steps = 27386/(batch_size*6)
    model.fit_generator(generate_arrays_from_file(batch_size, trainidx, img_list, tumor_list, tumorlines, liverlines, tumoridx, liveridx, minindex_list, maxindex_list),steps_per_epoch=steps,
                        epochs= 6000, verbose = 1, callbacks = [model_checkpoint], max_queue_size=10, workers=3, use_multiprocessing=True)

    print ('Finised Training .......')

if __name__ == '__main__':
    train_and_predict()