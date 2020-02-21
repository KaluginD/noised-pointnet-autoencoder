TF_INC=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
TF_NSYNC=~/Documents/experiments/pointnet-autoencoder/venv/lib/python2.7/site-packages/tensorflow/include/external/nsync/public
TF_LIB=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_lib())')

/usr/local/cuda-10.0/bin/nvcc  -std=c++11 -c -o tf_nndistance_g.cu.o tf_nndistance_g.cu -I $TF_INC -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC -O2
#/usr/local/cuda-10.0/bin/nvcc tf_nndistance_g.cu -o tf_nndistance_g.cu.o -c -O2 -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC
g++ -std=c++11 tf_nndistance.cpp tf_nndistance_g.cu.o -o tf_nndistance_so.so -shared -fPIC -I $TF_INC -I /usr/local/cuda-10.0/include -I $TF_NSYNC -lcudart -L /usr/local/cuda-10.0/lib64/ -L$TF_LIB -l:libtensorflow_framework.so.1 -O2 -D_GLIBCXX_USE_CXX11_ABI=0
#g++ -std=c++11 tf_nndistance.cpp tf_nndistance_g.cu.o -o tf_nndistance_so.so -shared -fPIC -I ~/Documents/experiments/pointnet-autoencoder/venv/lib/python2.7/site-packages/tensorflow/include -I /usr/local/cuda-10.0/include -I ~/Documents/experiments/pointnet-autoencoder/venv/lib/python2.7/site-packages/tensorflow/include/external/nsync/public -lcudart -L /usr/local/cuda-10.0/lib64/ -L/~/Documents/experiments/pointnet-autoencoder/venv/lib/python2.7/site-packages/tensorflow -ltensorflow_framework -O2 -D_GLIBCXX_USE_CXX11_ABI=0

