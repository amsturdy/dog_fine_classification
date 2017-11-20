import os, shutil, numpy as np

if __name__ == '__main__':
    this_dir=os.path.dirname(__file__)
    folder=os.path.join(this_dir,'../data/trainval')

    train_txt=open(os.path.join(this_dir,'../data/train.txt'),'r')
    lines1=train_txt.readlines()
    train_txt.close()
    val_txt=open(os.path.join(this_dir,'../data/val.txt'),'r')
    lines2=val_txt.readlines()
    val_txt.close()
    pictures_train={}
    for line in lines1:
	picture_name=line.split(' ')[0]+'.jpg'
	label=int(line.split(' ')[1][:-1])
	if not pictures_train.has_key(label):
	    pictures_train[label]=[picture_name]
	else:
	    pictures_train[label].append(picture_name)
    pictures_val={}
    for line in lines2:
	picture_name=line.split(' ')[0]+'.jpg'
	label=int(line.split(' ')[1][:-1])
	if not pictures_val.has_key(label):
	    pictures_val[label]=[picture_name]
	else:
	    pictures_val[label].append(picture_name)

    train_lmdb_txt=os.path.join(this_dir,'../data/train_lmdb.txt')
    test_lmdb_txt=os.path.join(this_dir,'../data/test_lmdb.txt')
    if os.path.exists(train_lmdb_txt):
	os.remove(train_lmdb_txt)
    if os.path.exists(test_lmdb_txt):
	os.remove(test_lmdb_txt)

    '''
    Class=np.arange(50)
    for i in np.arange(10)+40:
	Class[i]+=50
    merge=[]
    '''

    '''
    Class=np.arange(50)+50
    for i in np.arange(10)+40:
	Class[i]-=50
    merge=[]
    '''

    '''
    Class=np.arange(100)
    merge=[[3,46,58,59,68],[65,84],[30,44,54,82,89],[22,71],[25,98],[15,43],[37,62,64,81],
				[39,67,69,86],[34,74,78],[55,57],[9,11,88],[45,73],[47,49]]
    '''

    Class=[30,44,54,82,89]
    merge=[]

    total_merge=sum([len(i) for i in merge])
    new_label_maps_txt=open(os.path.join(this_dir,'../data/new_label_maps.txt'),'w')

    label_count=0
    for pre_label in Class:
	label=label_count
	is_merge=False
	for label_backward in range(len(merge)):
	    if is_merge:
		break
	    for merge_ in merge[label_backward]:
		if is_merge:
		    break
		if merge_==pre_label:
		    label=len(Class)-1+len(merge)-total_merge-label_backward
		    label_count-=1
		    is_merge=True

        new_label_maps_txt.write(str(pre_label)+':'+str(label)+'\n')
	'''
    	train_lmdb=open(train_lmdb_txt,'at')
    	test_lmdb=open(test_lmdb_txt,'at')
	pictures=pictures_train[pre_label]+pictures_val[pre_label]
	shuffle=np.random.permutation(len(pictures))
	for i in range(len(shuffle)):
	    if i<0.9*len(shuffle):
		train_lmdb.write(pictures[shuffle[i]]+' '+str(label)+'\n')
	    else:
		test_lmdb.write(pictures[shuffle[i]]+' '+str(label)+'\n')
	train_lmdb.close()
	test_lmdb.close()
	'''
	for phase in ['train','val']:
	    if phase == 'train':
    		txt=open(train_lmdb_txt,'at')
		pictures=pictures_train[pre_label]
	    else:
    		txt=open(test_lmdb_txt,'at')
		pictures=pictures_val[pre_label]
	    for i in pictures:
		txt.write(i+' '+str(label)+'\n')
	    txt.close()
	label_count+=1
		
    new_label_maps_txt.close()
