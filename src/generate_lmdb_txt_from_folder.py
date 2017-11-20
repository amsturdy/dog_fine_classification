import os, shutil, numpy as np

if __name__ == '__main__':
    this_dir=os.path.dirname(__file__)

    folder=os.path.join(this_dir,'../data/class_generate_folder/')

    train_lmdb_txt=os.path.join(this_dir,'../data/train_lmdb.txt')
    test_lmdb_txt=os.path.join(this_dir,'../data/test_lmdb.txt')

    if os.path.exists(train_lmdb_txt):
	os.remove(train_lmdb_txt)
    if os.path.exists(test_lmdb_txt):
	os.remove(test_lmdb_txt)

    new_label_maps_txt=open(os.path.join(this_dir,'../data/new_label_maps.txt'),'w')

    Class=[15,43]
    for label,pre_label in enumerate(Class):
        new_label_maps_txt.write(str(pre_label)+':'+str(label)+'\n')
	for phase in ['train','val']:
	    if phase=='train':
    		lmdb_txt=open(train_lmdb_txt,'at')
	    else:
    		lmdb_txt=open(test_lmdb_txt,'at')

            for read_dir, sub_dirs, files in os.walk(os.path.join(folder,phase,str(pre_label))):
                for file in files:
	            lmdb_txt.write(os.path.join(read_dir,file)+' '+str(label)+'\n')
	lmdb_txt.close()
    new_label_maps_txt.close()
