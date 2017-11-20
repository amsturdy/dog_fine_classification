import _init_paths, os, numpy as np
from cls import center_crop,mirror_crop,multi_crop,caffe_process,image_preprocess
import cv2, caffe

def label_map(label_maps_txt):
    f=open(label_maps_txt,'r')
    lines=f.readlines()
    f.close()
    label_maps={}
    for line in lines:
	first=int(line.split(':')[0])
	second=int(line.split(':')[1][:-1])
	if not label_maps.has_key(second):
	    label_maps[second]=[first]
	else:
	    label_maps[second].append(first)

    return label_maps

if __name__ == '__main__':
    this_dir=os.path.dirname(__file__)

    base_size = 256 # short size
    crop_size = 224
    # mean_value = np.array([128.0, 128.0, 128.0])  # BGR
    mean_value = np.array([104.0, 117.0, 123.0])  # BGR
    # std = np.array([128.0, 128.0, 128.0])  # BGR
    std = np.array([1.0, 1.0, 1.0])  # BGR
    crop_num = 12  # 1 and others for center(single)-crop, 12 for mirror(12)-crop, 144 for multi(144)-crop
    batch_size=1

    prob_layer = 'prob'
    gpu_id = 2 
    if not gpu_id==None:
        caffe.set_mode_gpu()
        caffe.set_device(gpu_id)
    else:
        caffe.set_mode_cpu()

    test_folder = 'data/test/image/'
    label_maps_txt=os.path.join(this_dir,'../data/label_maps.txt')
    new_label_maps_txt=os.path.join(this_dir,'../data/lmdb_75/new_label_maps.txt')
    result_txt=os.path.join(this_dir,'../results/result.txt')
    
    label_maps=label_map(label_maps_txt)
    new_label_maps=label_map(new_label_maps_txt)

    pictures={}
    for line in open(result_txt,'r'):
	pre_label=int(line.split('\t')[0])
	picture_id=line.split('\t')[1][:-1]
	if not pictures.has_key(pre_label):
	    pictures[pre_label]=[picture_id]
	else:
	    pictures[pre_label].append(picture_id)

    submit_txt=open(os.path.join(this_dir,'../results/submit.txt'),'w')
    for pre_label in pictures:
        class_num=len(new_label_maps[pre_label])
	if class_num==1:
	    for picture_id in pictures[pre_label]:
	        submit_txt.write(str(label_maps[new_label_maps[pre_label][0]][0])+'\t'+picture_id+'\n')
	else:
	    folder=''
	    for i in sorted(new_label_maps[pre_label]):
		folder+=(str(i)+'-')
	    folder=folder[:-1]
	    maps=label_map(os.path.join(this_dir,'../data/lmdb_'+folder, 'new_label_maps.txt'))
	    prototxt = os.path.join(this_dir,'../models/',folder,'jbaidu_resnet18_deploy.prototxt')
	    caffemodel=os.path.join(this_dir,'../weights/',folder,'jbaidu_resnet18_best.caffemodel')
	    net = caffe.Net(prototxt, caffemodel, caffe.TEST)
	    for picture_name in pictures[pre_label]:
	        full_name=os.path.join(test_folder,picture_name+'.jpg')
		_img = cv2.imread(full_name)
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
	        score_index = (score_vec / len(crops)).argsort()
		predict=score_index[-1]
	        submit_txt.write(str(label_maps[maps[predict][0]][0])+'\t'+picture_name+'\n')
    submit_txt.close()
