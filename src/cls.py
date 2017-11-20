import os,sys, _init_paths
#sys.path.append('~/caffe-master/python')
import numpy as np
import caffe
import cv2
from timer import Timer
import datetime

def image_preprocess(img,mean_value=[0.0,0.0,0.0],std=[1.0,1.0,1.0]):
    b, g, r = cv2.split(img)
    return cv2.merge([(b-mean_value[0])/std[0], (g-mean_value[1])/std[1], (r-mean_value[2])/std[2]])


def center_crop(img,crop_size): # single crop
    short_edge = min(img.shape[:2])
    if short_edge < crop_size:
        return
    yy = int((img.shape[0] - crop_size) / 2)
    xx = int((img.shape[1] - crop_size) / 2)
    return img[yy: yy + crop_size, xx: xx + crop_size]


def over_sample(img,crop_size):  # 12 crops of image
    short_edge = min(img.shape[:2])
    if short_edge < crop_size:
        return
    yy = int((img.shape[0] - crop_size) / 2)
    xx = int((img.shape[1] - crop_size) / 2)
    sample_list = [img[:crop_size, :crop_size], img[-crop_size:, -crop_size:], img[:crop_size, -crop_size:],
                   img[-crop_size:, :crop_size], img[yy: yy + crop_size, xx: xx + crop_size],
                   cv2.resize(img, (crop_size, crop_size))]
    return sample_list


def mirror_crop(img,base_size,crop_size):  # 12*len(size_list) crops
    crop_list = []
    img_resize = cv2.resize(img, (base_size, base_size))
    mirror = img_resize[:, ::-1]
    crop_list.extend(over_sample(img_resize,crop_size))
    crop_list.extend(over_sample(mirror,crop_size))
    return crop_list


def multi_crop(img,crop_size):  # 144(12*12) crops
    crop_list = []
    size_list = [256, 288, 320, 352]  # crop_size: 224
    # size_list = [270, 300, 330, 360]  # crop_size: 235
    # size_list = [320, 352, 384, 416]  # crop_size: 299
    # size_list = [352, 384, 416, 448]  # crop_size: 320
    short_edge = min(img.shape[:2])
    for i in size_list:
        img_resize = cv2.resize(img, (img.shape[1] * i / short_edge, img.shape[0] * i / short_edge))
        yy = int((img_resize.shape[0] - i) / 2)
        xx = int((img_resize.shape[1] - i) / 2)
        for j in xrange(3):
            left_center_right = img_resize[yy * j: yy * j + i, xx * j: xx * j + i]
            mirror = left_center_right[:, ::-1]
            crop_list.extend(over_sample(left_center_right),crop_size)
            crop_list.extend(over_sample(mirror),crop_size)
    return crop_list


def caffe_process(net,_input,prob_layer):
    _input = _input.transpose(0, 3, 1, 2)
    net.blobs['data'].reshape(*_input.shape)
    net.blobs['data'].data[...] = _input
    net.forward()

    return np.sum(net.blobs[prob_layer].data, axis=0)


if __name__ == '__main__':
    test_folder = 'data/test/image/'
    result_file = 'results/result.txt'
    base_size = 256 # short size
    crop_size = 224
    # mean_value = np.array([128.0, 128.0, 128.0])  # BGR
    mean_value = np.array([104.0, 117.0, 123.0])  # BGR
    # std = np.array([128.0, 128.0, 128.0])  # BGR
    std = np.array([1.0, 1.0, 1.0])  # BGR
    crop_num = 12  # 1 and others for center(single)-crop, 12 for mirror(12)-crop, 144 for multi(144)-crop
    class_num=75
    batch_size=1

    gpu_id = 2 
    model_weights = 'weights/resnet50_75/jbaidu_ResNet_50_75_iter_12000.caffemodel'
    model_deploy = 'models/resnet50_75/deploy.prototxt'
    prob_layer = 'prob'
    if not gpu_id==None:
        caffe.set_mode_gpu()
        caffe.set_device(gpu_id)
    else:
        caffe.set_mode_cpu()
    net = caffe.Net(model_deploy, model_weights, caffe.TEST)
    f=open(result_file,'w')
    print '\nDetect begin: \n'
    timer=Timer()
    timer.tic()
    for file in sorted(os.listdir(test_folder)):
        _img = cv2.imread(os.path.join(test_folder,file))
        _img = cv2.resize(_img, (int(_img.shape[1] * base_size / min(_img.shape[:2])),
                                 int(_img.shape[0] * base_size / min(_img.shape[:2])))
                          )
        _img = image_preprocess(_img,mean_value,std)

        score_vec = np.zeros(class_num, dtype=np.float32)
        crops = []
        if crop_num == 1:
            crops.append(center_crop(_img,crop_size))
        elif crop_num == 12:
            crops.extend(mirror_crop(_img,base_size,crop_size))
        elif crop_num == 144:
            crops.extend(multi_crop(_img,crop_size))
        else:
            crops.append(center_crop(_img,crop_size))

        iter_num = int(len(crops) / batch_size)
        for j in xrange(iter_num):
            score_vec += caffe_process(net,np.asarray(crops, dtype=np.float32)[j*batch_size:(j+1)*batch_size],prob_layer)
        score_index = (-score_vec / len(crops)).argsort()
	predict=score_index[0]
	f.write(str(predict)+'\t'+file.split('.')[0]+'\n')
    timer.toc()
    print ('Classification took {:.3f}s').format(timer.total_time)
    print '\nDetect begin: \n'
    f.close()
