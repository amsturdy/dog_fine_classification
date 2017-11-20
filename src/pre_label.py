import os, sys, shutil, numpy as np

if __name__ == '__main__':
    this_dir=os.path.dirname(__file__)

    class_data=os.path.join(this_dir,'../data/class_data')
    copy=False
    if not os.path.exists(class_data):
	copy=True
 
    train=open(os.path.join(this_dir,'../data/trainval/data_train_image.txt'),'r')
    lines1=train.readlines()
    train.close()
    val=open(os.path.join(this_dir,'../data/trainval/val.txt'),'r')
    lines2=val.readlines()
    val.close()

    current_label=0
    label_maps={}
    picture_labels={}
    mul_labels={}
    label_maps_txt=open(os.path.join(this_dir,'../data/label_maps.txt'),'w')

    train_txt=open(os.path.join(this_dir,'../data/train.txt'),'w')
    for line in lines1:
        item=line.split(' ')
        picture_name=item[0]
	full_picture_name='trainval/train/'+picture_name+'.jpg'
        label=int(item[1])
	if not label_maps.has_key(label):
	    label_maps[label]=current_label
	    label_maps_txt.write(str(label)+':'+str(current_label)+'\n')
	    current_label+=1
	if copy:
	    current_label_folder=os.path.join(class_data,'train', str(label_maps[label]))
	    if not os.path.exists(current_label_folder):
		os.makedirs(current_label_folder)
	    shutil.copy( os.path.join(this_dir,'../data/'+full_picture_name), 
								current_label_folder)
        if not picture_labels.has_key(picture_name):
            picture_labels[picture_name]=[label_maps[label]]
	    train_txt.write(full_picture_name+' '+str(label_maps[label])+'\n')
        else:
	    picture_labels[picture_name].append(label_maps[label])
	    mul_labels[picture_name]=picture_labels[picture_name]
    train_txt.close()

    val_txt=open(os.path.join(this_dir,'../data/val.txt'),'w')
    for line in lines2:
        item=line.split(' ')
        picture_name=item[0]
	full_picture_name='trainval/test1/'+picture_name+'.jpg'
        label=int(item[1])
	if not label_maps.has_key(label):
	    label_maps[label]=current_label
	    label_maps_txt.write(str(label)+':'+str(current_label)+'\n')
	    current_label+=1
	if copy:
	    current_label_folder=os.path.join(class_data,'val', str(label_maps[label]))
	    if not os.path.exists(current_label_folder):
		os.makedirs(current_label_folder)
	    shutil.copy( os.path.join(this_dir,'../data/'+full_picture_name), 
								current_label_folder)
        if not picture_labels.has_key(picture_name):
            picture_labels[picture_name]=label_maps[label]
	    val_txt.write(full_picture_name+' '+str(label_maps[label])+'\n')
        else:
	    picture_labels[picture_name].append(label_maps[label])
	    mul_labels[picture_name]=picture_labels[picture_name]
    val_txt.close()

    label_maps_txt.close()
    print label_maps
    print mul_labels
    mul_picture_name=[i for i in mul_labels]
    mul_min=[min(mul_labels[i]) for i in mul_picture_name]
    mul_argsort=np.argsort(np.array(mul_min))
    f=open(os.path.join(this_dir,'../data/mul_labels.txt'),'w')
    for i in mul_argsort:
	f.write(mul_picture_name[i])
	for j in sorted(mul_labels[mul_picture_name[i]]):
	    f.write(' '+str(j))
	f.write('\n')
    f.close()

