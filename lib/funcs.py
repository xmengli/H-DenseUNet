import numpy as np
from keras import backend as K
from skimage import measure
def predict_tumor_inwindow(model, imgs_test, num, mini, maxi, args):

    batch = args.b
    img_deps = args.input_size
    img_rows = args.input_size
    img_cols = args.input_cols


    window_cols = (img_cols/4)
    count = 0
    box_test = np.zeros((batch,img_deps,img_rows,img_cols, 1), dtype="float32")

    x = imgs_test.shape[0]
    y = imgs_test.shape[1]
    z = imgs_test.shape[2]
    right_cols = int(min(z,maxi[2]+10)-img_cols)
    left_cols  = max(0,min(mini[2]-5, right_cols))
    score = np.zeros((x, y, z, num), dtype= 'float32')
    score_num = np.zeros((x, y, z, num), dtype= 'int16')

    for cols in xrange(left_cols,right_cols+window_cols,window_cols):
        # print ('and', z-img_cols,z)
        if cols > z - img_cols:
            patch_test = imgs_test[0:img_deps, 0:img_rows, z-img_cols:z]
            box_test[count, :, :, :, 0] = patch_test
            # print ('final', img_cols-window_cols, img_cols)
            patch_test_mask = model.predict(box_test, batch_size=batch, verbose=0)
            patch_test_mask = K.softmax(patch_test_mask)
            patch_test_mask = K.eval(patch_test_mask)
            patch_test_mask = patch_test_mask[:,:,:,1:-1,:]

            for i in xrange(batch):
                score[0:img_deps, 0:img_rows,  z-img_cols+1:z-1, :] += patch_test_mask[i]
                score_num[0:img_deps, 0:img_rows,  z-img_cols+1:z-1, :] += 1
        else:
            patch_test = imgs_test[0:img_deps, 0:img_rows, cols:cols + img_cols]
            box_test[count, :, :, :, 0] = patch_test
            patch_test_mask = model.predict(box_test, batch_size=batch, verbose=0)
            patch_test_mask = K.softmax(patch_test_mask)
            patch_test_mask = K.eval(patch_test_mask)
            patch_test_mask = patch_test_mask[:,:,:,1:-1,:]
            for i in xrange(batch):
                score[0:img_deps, 0:img_rows, cols+1:cols+img_cols-1, :] += patch_test_mask[i]
                score_num[0:img_deps, 0:img_rows, cols+1:cols+img_cols-1, :] += 1
    score = score/(score_num+1e-4)
    score1 = score[:,:,:,num-2]
    score2 = score[:,:,:,num-1]
    return score1, score2


def predict_window_mulgpu(model,batch, imgs_test, img_deps, img_rows, img_cols, multiloss):

    window_deps = (img_deps/3)*2
    window_rows = (img_rows/3)*2
    window_cols = (img_cols/3)*2

    current_test = imgs_test
    x = current_test.shape[0]
    y = current_test.shape[1]
    z = current_test.shape[2]
    score = np.zeros((x,y,z,2), dtype= 'float32')
    score_num = np.zeros((x,y,z,2), dtype= 'int16')

    count = 0
    deplist = []
    rowlist = []
    collist = []
    num = 0

    box_test = np.zeros((batch,img_deps,img_rows,img_cols,1), dtype="float32")
    for deps in xrange(0,x-img_deps+window_deps,window_deps):
        print (deps)
        for rows in xrange(0, y-img_rows+window_rows, window_rows):
            for cols in xrange(0,z-img_cols+window_cols,window_cols):
                if deps>x-img_deps:
                    deps = x-img_deps
                elif rows > y-img_rows:
                    rows = y-img_rows
                elif cols>z-img_cols:
                    cols = z-img_cols
                elif deps>x-img_deps and rows > y - img_rows:
                    deps = x - img_deps
                    rows = y - img_rows
                elif deps>x-img_deps and cols > z - img_cols:
                    deps = x - img_deps
                    cols = z - img_cols
                elif rows>y-img_rows and cols > z-img_cols:
                    rows = y - img_rows
                    cols = z - img_cols
                elif rows>y-img_rows and cols > z-img_cols and deps > x-img_deps:
                    deps = x - img_deps
                    rows = y - img_rows
                    cols = z - img_cols
                if count == batch:
                    count = 0
                    deplist = []
                    rowlist = []
                    collist = []
                    box_test = np.zeros((batch, img_deps, img_rows, img_cols, 1), dtype="float32")
                patch_test = current_test[deps:deps+img_deps, rows:rows+img_rows, cols:cols+img_cols]
                deplist.append(deps)
                rowlist.append(rows)
                collist.append(cols)
                box_test[count,:,:,:,0] = patch_test
                count += 1
                del patch_test
                if count == batch:
                    num = num+1
                    print ('num: ',num)
                    print ('box:', box_test.shape)

                    patch_test_mask = model.predict(box_test, verbose=0)

                    if multiloss:
                        patch_test_mask = patch_test_mask[2]
                    patch_test_mask = K.softmax(patch_test_mask)
                    patch_test_mask = K.eval(patch_test_mask)
                    print ('predict finish')
                    for i in xrange(batch):
                        score[deplist[i]:deplist[i]+img_deps, rowlist[i]:rowlist[i]+img_rows, collist[i]:collist[i]+img_cols,:] += patch_test_mask[i]
                        score_num[deplist[i]:deplist[i]+img_deps, rowlist[i]:rowlist[i]+img_rows, collist[i]:collist[i]+img_cols,:] += 1
                    # print ('queue finish')
                    del box_test, patch_test_mask, deplist, rowlist, collist
    score = score / (score_num)
    score2 = score[:,:,:,1]
    return score2

def get_binary_mask(score, id):
    ## load affine
    # label, header = load('/home/xmli/Data_gpu7/NewThresTestData/test-volume-' + str(id) + '.nii')
    Segmask = GeneSeglivertumor(score)
    Segmask = np.int16(Segmask)
    return Segmask

def GeneSeglivertumor(score):

    score[score>=0.5] = 1
    score[score<0.5] = 0
    box = []
    [liver_labels, num] = measure.label(score, return_num = True)
    region = measure.regionprops(liver_labels)
    for i in xrange(num):
        box.append(region[i].area)
    label_num = box.index(max(box))+1
    liver_labels[liver_labels!=label_num] = 0
    liver_labels[liver_labels==label_num] = 1

    # labels = ndimage.binary_fill_holes(score).astype(int)
    # labels = score
    return liver_labels
