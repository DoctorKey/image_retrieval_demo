import caffe
import numpy as np


LABELS = None

def get_transformer(data_shape, caffe_root):
    # load the mean ImageNet image (as distributed with Caffe) for subtraction
    mu = np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy')
    mu = mu.mean(1).mean(1)  # average over pixels to obtain the mean (BGR) pixel values
    # print 'mean-subtracted values:', zip('BGR', mu)

    # create transformer for the input called 'data'
    transformer = caffe.io.Transformer({'data': data_shape})

    transformer.set_transpose('data', (2, 0, 1))  # move image channels to outermost dimension
    transformer.set_mean('data', mu)  # subtract the dataset-mean value in each channel
    transformer.set_raw_scale('data', 255)  # rescale from [0, 1] to [0, 255]
    transformer.set_channel_swap('data', (2, 1, 0))  # swap channels from RGB to BGR
    return transformer



def decode_predictions(output_prob, top=5):
    global LABELS
    if len(output_prob.shape) != 2 or output_prob.shape[1] != 1000:
        raise ValueError('`decode_predictions` expects '
                         'a batch of predictions '
                         '(i.e. a 2D array of shape (samples, 1000)). '
                         'Found array with shape: ' + str(output_prob.shape))
    if LABELS is None:
        labels_file = caffe_root + 'data/ilsvrc12/synset_words.txt'
        LABELS = np.loadtxt(labels_file, str, delimiter='\t')
    # sort top five predictions from softmax output
    top_inds = output_prob.argsort()[::-1][:top]  # reverse sort and take five largest items
    results = zip(output_prob[top_inds], LABELS[top_inds])
    return results
