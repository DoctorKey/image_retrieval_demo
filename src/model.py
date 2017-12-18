# -*- coding: utf-8 -*-

import caffe
import os
import subprocess

class Model(object):

    def __init__(self, caffe_root):

        self.caffe_root = caffe_root
        self.model_def = caffe_root + 'models/bvlc_reference_caffenet/deploy.prototxt'
        self.model_weights = caffe_root + 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'

        self.net = None

    def model_init(self):

        if os.path.isfile(self.model_weights):
            print 'CaffeNet found.'
        else:
            print 'Downloading pre-trained CaffeNet model...'
            download_file = self.caffe_root + 'scripts/download_model_binary.py'
            folder = self.caffe_root + 'models/bvlc_reference_caffenet'
            subprocess.call([download_file, folder])

        self.net = caffe.Net(self.model_def,      # defines the structure of the model
                            self.model_weights,  # contains the trained weights
                            caffe.TEST)     # use test mode (e.g., don't perform dropout)

        # set the size of the input (we can skip this if we're happy
        #  with the default; we can also change it later, e.g., for different batch sizes)
        self.net.blobs['data'].reshape(50,  # batch size
                                  3,  # 3-channel (BGR) images
                                  227, 227)  # image size is 227x227

    def get_input_datashape(self):
        return self.net.blobs['data'].data.shape


    def get_predict(self, transformed_image):
        # copy the image data into the memory allocated for the net
        self.net.blobs['data'].data[...] = transformed_image
        ### perform classification
        output = self.net.forward()
        output_prob = output['prob'][0]  # the output probability vector for the first image in the batch
        return output_prob


    def get_feature(self, transformed_image):
        # copy the image data into the memory allocated for the net
        self.net.blobs['data'].data[...] = transformed_image
        ### perform classification
        output = self.net.forward()
        feat = self.net.blobs['fc6'].data[0]
        return feat


