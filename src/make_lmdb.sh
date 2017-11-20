
lmdb_path=data/lmdb
height=312
width=312
folder=data/

if [ ! -d "$lmdb_path" ];then
    mkdir $lmdb_path
else
    echo "The path $lmdb_path has existed, please change the path to lmdb!"
    exit 1
fi

#data/train_lmdb.txt \
echo $"Create train_lmdb\n"
caffe-master/build/tools/convert_imageset --shuffle --resize_height=$height --resize_width=$width \
$folder \
data/train.txt \
$lmdb_path/train_lmdb

if [ $? -ne 0 ];then
    echo "Create train_lmdb failed!"
    exit 1
fi

#data/test_lmdb.txt \
echo $"Create test_lmdb\n"
caffe-master/build/tools/convert_imageset --shuffle --resize_height=$height --resize_width=$width \
$folder \
data/val.txt \
$lmdb_path/test_lmdb

if [ $? -ne 0 ];then
    echo "Create test_lmdb failed!"
    exit 1
fi

#mv data/train_lmdb.txt data/test_lmdb.txt data/new_label_maps.txt $lmdb_path
echo $"Make $lmdb done\n"

#caffe-master/build/tools/caffe train -solver solvers/resnet50_75/jbaidu_ResNet_50_solver.prototxt -weights weights/ResNet_50_model.caffemodel -gpu 2 >logs/resnet50_75/log.txt 2>&1
