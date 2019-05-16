"""Test ImageNet pretrained DenseNet"""
from __future__ import print_function
import sys
sys.path.insert(0,'Keras-2.0.8')
from multiprocessing.dummy import Pool as ThreadPool
from medpy.io import load
import numpy as np
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint
from hybridnet import dense_rnn_net
from denseunet3d import denseunet_3d
import keras.backend as K
import os
import time
from loss import weighted_crossentropy
from skimage.transform import resize
import argparse
import os

K.set_image_dim_ordering('tf')

#  global parameters
parser = argparse.ArgumentParser(description='Keras DenseUnet Training')
#  data folder
parser.add_argument('-data', type=str, default='data/', help='test images')
parser.add_argument('-save_path', type=str, default='Experiments/')
#  other paras
parser.add_argument('-b', type=int, default=1)
parser.add_argument('-input_size', type=int, default=224)
parser.add_argument('-model_weight', type=str, default='./model/model_best.hdf5')
parser.add_argument('-input_cols', type=int, default=8)
parser.add_argument('-arch', type=str, default='')

#  data augment
parser.add_argument('-mean', type=int, default=48)
args = parser.parse_args()

thread_num = 14
liverlist = [32,34,38,41,47,87,89,91,105,106,114,115,119]
def load_seq_crop_data_masktumor_try(Parameter_List):
    img = Parameter_List[0]
    tumor = Parameter_List[1]
    lines = Parameter_List[2]
    numid = Parameter_List[3]
    minindex = Parameter_List[4]
    maxindex = Parameter_List[5]
    #  randomly scale
    scale = np.random.uniform(0.8,1.2)
    deps = int(args.input_size * scale)
    rows = int(args.input_size * scale)
    cols = args.input_cols

    sed = np.random.randint(1,numid)
    cen = lines[sed-1]
    cen = np.fromstring(cen, dtype=int, sep=' ')
    # print (cen)
    a = min(max(minindex[0] + deps/2, cen[0]), maxindex[0]- deps/2-1)
    b = min(max(minindex[1] + rows/2, cen[1]), maxindex[1]- rows/2-1)
    c = min(max(minindex[2] + cols/2, cen[2]), maxindex[2]- cols/2-1)
    cropp_img = img[a - deps / 2:a + deps / 2, b - rows / 2:b + rows / 2,
                c - args.input_cols / 2: c + args.input_cols / 2].copy()
    cropp_tumor = tumor[a - deps / 2:a + deps / 2, b - rows / 2:b + rows / 2,
                  c - args.input_cols / 2:c + args.input_cols / 2].copy()

    cropp_img -= args.mean
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
    cropp_tumor = resize(cropp_tumor, (args.input_size,args.input_size, args.input_cols), order=0, mode='edge', cval=0, clip=True, preserve_range=True)
    cropp_img   = resize(cropp_img, (args.input_size,args.input_size, args.input_cols), order=3, mode='constant', cval=0, clip=True, preserve_range=True)
    return cropp_img, cropp_tumor


def generate_arrays_from_file(batch_size, trainidx, img_list, tumor_list, tumorlines, liverlines, tumoridx, liveridx, minindex_list, maxindex_list):
    while 1:
        X = np.zeros((batch_size, args.input_size, args.input_size, args.input_cols,1), dtype='float32')
        Y = np.zeros((batch_size, args.input_size, args.input_size, args.input_cols,1), dtype='int16')
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

def train_and_predict(args):

    print('-'*30)
    print('Creating and compiling model...')
    print('-'*30)

    if args.arch == "3dpart":
        model = denseunet_3d(args)
        model_path = "/3dpart_model"
        sgd = SGD(lr=1e-3, momentum=0.9, nesterov=True)
        model.compile(optimizer=sgd, loss=[weighted_crossentropy])
        model.load_weights(args.model_weight, by_name=True, by_gpu=True, two_model=True, by_flag=True)
    else:
        model = dense_rnn_net(args)
        model_path = "/hybrid_model"
        sgd = SGD(lr=1e-3, momentum=0.9, nesterov=True)
        model.compile(optimizer=sgd, loss=[weighted_crossentropy])
        model.load_weights(args.model_weight)

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
    for idx in xrange(131):
        img, img_header = load(args.data + '/myTrainingData/volume-' + str(idx) + '.nii' )
        tumor, tumor_header = load(args.data + '/myTrainingData/segmentation-' + str(idx) + '.nii')
        img_list.append(img)
        tumor_list.append(tumor)

        maxmin = np.loadtxt(args.data+'/myTrainingDataTxt/LiverBox/box_' + str(idx) + '.txt', delimiter=' ')
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

        f1 = open(args.data+ '/myTrainingDataTxt/TumorPixels/tumor_' + str(idx) + '.txt','r')
        tumorline = f1.readlines()
        tumorlines.append(tumorline)
        tumoridx.append(len(tumorline))
        f1.close()

        f2 = open(args.data+ '/myTrainingDataTxt/LiverPixels/liver_' + str(idx) + '.txt','r')
        liverline = f2.readlines()
        liverlines.append(liverline)
        liveridx.append(len(liverline))
        f2.close()

    if not os.path.exists(args.save_path +model_path):
        os.mkdir(args.save_path + model_path)
    if not os.path.exists(args.save_path + "/history"):
        os.mkdir(args.save_path + '/history')
    else:
        if os.path.exists(args.save_path + "/history/lossbatch.txt"):
            os.remove(args.save_path + '/history/lossbatch.txt')
        if os.path.exists(args.save_path + "/history/lossepoch.txt"):
            os.remove(args.save_path + '/history/lossepoch.txt')
    model_checkpoint = ModelCheckpoint(args.save_path + model_path+'/weights.{epoch:02d}-{loss:.2f}.hdf5', monitor='loss', verbose = 1,
                                       save_best_only=False,save_weights_only=False,mode = 'min', period = 1)
    print('-'*30)
    print('Fitting model......')
    print('-'*30)
    steps = 27386 / (args.b * 6)
    model.fit_generator(generate_arrays_from_file(args.b, trainidx, img_list, tumor_list, tumorlines, liverlines,
                                                  tumoridx, liveridx, minindex_list, maxindex_list),
                        steps_per_epoch=steps,
                        epochs= 6000, verbose = 1, callbacks = [model_checkpoint], max_queue_size=10, workers=3, use_multiprocessing=True)
    print ('Finised Training .......')

if __name__ == '__main__':
    train_and_predict(args)
