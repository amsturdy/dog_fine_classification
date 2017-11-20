#!/usr/bin/env python

import os, sys, shutil, cv2, h5py, _init_paths
import matplotlib.pyplot as plt, numpy as np, scipy.io as sio
import caffe, pdb
from timer import Timer

def feature_maps_to_h5(image_dir, prototxt, caffemodel, h5_path):
    gpu_id=3
    if gpu_id==None:
        caffe.set_mode_cpu()
    else:
      	caffe.set_mode_gpu()
       	caffe.set_device(gpu_id)
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)

    resize=(299, 299)
    crop_size=(224, 224)
    im_mean=(104,117,123)

    h5=open(h5_path,'w')
    data=[]
    labels=[]
	shuffle=np.random.permutation(len(lines))
    for im_name in sorted(os.listdir(image_dir)):
    	full_name=os.path.join(image_dir,im_name)		
	    im=cv2.imread(full_name)
	    crop_ims=pre_image(im, resize, crop_size, im_mean)

        blob = np.zeros((10, crop_size[0], crop_size[1], 3),dtype=np.float32)
	    blob=crop_ims
	    blob=blob.transpose((0,3,1,2))
	    net.blobs['data'].data[...] = blob
        out = net1.forward()
	    feature=net.blobs['pool5'].data

	    labels_10=label*np.ones((10,1))
	    data.append(features)
	    labels.append(labels_10)

	data=np.row_stack(data)
    labels=np.row_stack(labels)
    with h5py.File(h5,'w') as h:
	    h.create_dataset('data',data=data)
	    h.create_dataset('label',data=labels)
