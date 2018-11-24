from keras.layers import concatenate
from keras.layers import Lambda
from keras.models import Model

import tensorflow as tf

def make_parallel(model, gpu_count, mini_batch):
    def get_slice(data, idx, parts):
        shape = tf.shape(data)
        # print ("data",data)
        # print ("shape",shape[:1])
        # print (shape[1:])
        
        # size = tf.concat([ shape[:1] // parts, shape[1:] ],axis=0)
        # print (size)
        # print ('1',shape[:1] // parts)
        # print ('2',shape[1:]*0)
        # stride = tf.concat([ shape[:1] // parts, shape[1:]*0 ],axis=0)
        # print (stride)
        # start = stride * idx
        # print (start)
        # print ('return',tf.slice(data,start,size))
        # # exit(0)
        # print ('idx', idx*mini_batch,(idx+1)*mini_batch )
        return data[idx*mini_batch:(idx+1)*mini_batch,:, :,:]
        # data[25:50, :, :, :]
        # return tf.slice(data, start, size)

    outputs_all = []
    for i in range(len(model.outputs)):
        outputs_all.append([])
    # print (outputs_all)
    #Place a copy of the model on each GPU, each getting a slice of the batch

    for i in range(gpu_count):
        id = i
        # print ('loading'+str(id))
        with tf.device('/gpu:%d' % id):
            with tf.name_scope('tower_%d' % i) as scope:
                inputs = []
                # print ('ssssssssssss')
                # print ('rr',model.inputs)
                #Slice each input into a piece for processing on this GPU
                for x in model.inputs:
                    # print ('x', x)
                    input_shape = tuple(x.get_shape().as_list())[1:]
                    # print (input_shape)
                    slice_n = Lambda(get_slice, output_shape=input_shape, arguments={'idx':i,'parts':gpu_count})(x)
                    # print ('slice_n', slice_n)
                    inputs.append(slice_n)

                # print ('ii',inputs)
                outputs = model(inputs)
                # print ('xx',outputs)

                # print ('ssdadsa')
                if not isinstance(outputs, list):
                    outputs = [outputs]
                # print ('ssd')
                #Save all the outputs for merging back together later
                for l in range(len(outputs)):
                    outputs_all[l].append(outputs[l])
                # print ('hard')
    # merge outputs on CPU
    with tf.device('/cpu:0'):
        merged = []
        for outputs in outputs_all:
            merged.append(concatenate(outputs, axis=0))
        return Model( outputs=merged, inputs= model.inputs)


