"""Test ImageNet pretrained DenseNet"""
from __future__ import print_function
import sys
sys.path.insert(0,'/home/xmli/big/livertumor/Keras-2.0.8')
sys.path.insert(0,'/home/xmli/big/livertumor/mylib')
from multiprocessing.dummy import Pool as ThreadPool
from medpy.io import load
import numpy as np
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint
from net import dense_rnn_net
import keras.backend as K
import os
import time
from skimage.transform import resize
import argparse

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
K.set_image_dim_ordering('tf')

#  global parameters
parser = argparse.ArgumentParser(description='Keras DenseUnet Training')
#  data folder
parser.add_argument('-data', type=str, default='/home/xmli/gpu7_xmli/LiverChallengeData/myTestData/test-volume-', help='test images')
parser.add_argument('-save_path', type=str, default='results')
#  other paras
parser.add_argument('-b', type=int, default=1)
parser.add_argument('-input_size', type=int, default=224)
parser.add_argument('-model_weight', type=str, default='./model/model_best.hdf5')
parser.add_argument('-input_cols', type=int, default=8)

#  data augment
parser.add_argument('-mean', type=int, default=48)
args = parser.parse_args()


path = './result_train_dense167_fast_bauto3d_residualconv_endtoend_big/'
batch_size = 1
img_deps = 224
img_rows = 224
img_cols = 8
thread_num = 14
txtfile = 'myTrainingDataTxt'
mean = 48

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

def train_and_predict(args):

    print('-'*30)
    print('Creating and compiling model...')
    print('-'*30)
    model = dense_rnn_net(args)
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
    model.fit_generator(generate_arrays_from_file(batch_size, trainidx, img_list, tumor_list, tumorlines, liverlines,
                                                  tumoridx, liveridx, minindex_list, maxindex_list),
                        steps_per_epoch=steps,
                        epochs= 6000, verbose = 1, callbacks = [model_checkpoint], max_queue_size=10, workers=3, use_multiprocessing=True)
    print ('Finised Training .......')

if __name__ == '__main__':
    train_and_predict(args)