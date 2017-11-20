import _init.paths
import os, shutil, numpy as np
import caffe

def read_txt(filename):
    f=open(filename,'r')
    pictures={}
    for line in f:
	picture_name=line.split(' ')[0]+'.jpg'
	label=int(line.split(' ')[1][:-1])
	if not pictures_train.has_key(label):
	    pictures[label]=[picture_name]
	else:
	    pictures[label].append(picture_name)
    f.close()
    return pictures

class MyDataLayer(caffe.Layer):
    def _shuffle_class(self):
	self._class=np.random.permutation(np.arange(len(self._pictures)))
	self._class_inds=0

    def _shuffle_img(self,class_id):
	self._img[class_id]=np.random.permutation(np.arange(len(self._pictures[class_id])))
	self._img_inds[class_id]=0

    def _shuffle_init(self):
	self._shuffle_class()
	self._img_inds=np.zeros(len(pictures),int)
	self._img=[]
	for class_id in np.arange(len(self._pictures)):
	    self._img.append(np.random.permutation(np.arange(len(self._pictures[class_id]))))

    def _get_next_minibatch(self):
	blobs={'data':np.zeros((self._batchsize,3,self._resize,self._resize),float),'label':np.zeros((self._batchsize,1,1,1),int)}
	if self._class_inds+self._batchsize >= len(pictures):
	    self._shuffle_class()
	for i in range(self._batchsize):
	    label=self._class[self._class_inds]
	    if self._img_inds[label] >= len(pictures[label]):
		self._shuffle_img(label)
	    img_name=self._pictures[label][self._img[label][self._img_inds[label]]]
	    self._class_inds+=1
	    self._img_inds[label]+=1
	    img=cv2.imread(img_name)
	    img=cv2.resize(img,self._resize,self._resize)
	    blobs['data'][i]=im.transpose((2,0,1))
	    blobs['label'][i,...]=label
	return blobs

    def setup(self, bottom, top):
        """Setup the RoIDataLayer."""

        # parse the layer parameter string, which must be valid YAML
        layer_params = yaml.load(self.param_str)

        self._dir = os.path.dirname(__file__)
	self._pictures=read_txt(os.path.join(self._dir,'../data/train.txt'))
	self._batchsize=16
	self._shuffle_init()
        self._name_to_top_map = {}

        # data blob: holds a batch of N images, each with 3 channels
        idx = 0
        top[idx].reshape(self._batchsize, 3, self._resize, self._resize)
        self._name_to_top_map['data'] = idx
	idx += 1
        top[idx].reshape(self._batchsize, 1)
        self._name_to_top_map['label'] = idx

        print 'RoiDataLayer: name_to_top:', self._name_to_top_map
        assert len(top) == len(self._name_to_top_map)

    def forward(self, bottom, top):
        """Get blobs and copy them into this layer's top blob vector."""
        blobs = self._get_next_minibatch()

        for blob_name, blob in blobs.iteritems():
            top_ind = self._name_to_top_map[blob_name]
            shape = blob.shape
            top[top_ind].reshape(*(blob.shape))
            # Copy data into net's input blobs
            top[top_ind].data[...] = blob.astype(np.float32, copy=False)

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass

