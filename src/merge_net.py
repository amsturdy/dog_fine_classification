#!/usr/bin/env python

import os, sys, shutil, matplotlib.pyplot as plt, numpy as np, _init_paths
import cv2, scipy.io as sio, pdb
import caffe

if __name__ == '__main__':
    this_dir=os.path.dirname(__file__)

    prototxt1 = os.path.join(this_dir,'../models/inception_v4-resnetv2/jbaidu_inception_v4-resnetv2_deploy.prototxt')
    caffemodel1 = os.path.join(this_dir,'../weights/inception_v4-resnetv2/jbaidu_inception_v4-resnetv2_iter_0.caffemodel')
    prototxt2 = os.path.join(this_dir,'../models/deploy_inception-v4.prototxt')
    caffemodel2 = os.path.join(this_dir,'../weights/inception-v4.caffemodel')
    prototxt3 = os.path.join(this_dir,'../models/deploy_inception-resnet-v2.prototxt')
    caffemodel3 = os.path.join(this_dir,'../weights/inception-resnet-v2.caffemodel')

    gpu_id=3
    if gpu_id==None:
        caffe.set_mode_cpu()
    else:
      	caffe.set_mode_gpu()
       	caffe.set_device(gpu_id)
    net1 = caffe.Net(prototxt1, caffemodel1, caffe.TEST)
    net2 = caffe.Net(prototxt2, caffemodel2, caffe.TEST)
    net3 = caffe.Net(prototxt3, caffemodel3, caffe.TEST)

    for layer_name, param in net1.params.iteritems():
	if layer_name[:2]=='m_':
	    if net2.params.has_key(layer_name[2:]):
	        for i in range(len(param)):
		    param[i].data[...]=net2.params[layer_name[2:]][i].data[...]
	    else:
		print layer_name
	else:
	    if net3.params.has_key(layer_name):
	        for i in range(len(param)):
		    param[i].data[...]=net3.params[layer_name][i].data[...]
	    else:
	        print layer_name

    net1.save(os.path.join(this_dir,'../weights/inception_v4-resnetv2/inception_v4-resnetv2.caffemodel'))
