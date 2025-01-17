# sudo docker build -t mask_rcnn_v1 -f Dockerfile .

# TODO:
# pip3 install --upgrade pip
# pip3 install imgaug
# pip install numpy==1.17.0

FROM nvidia/cuda:11.3.1-devel-ubuntu16.04

# Supress warnings about missing front-end. As recommended at:
# http://stackoverflow.com/questions/22466255/is-it-possibe-to-answer-dialog-questions-when-installing-under-docker
ARG DEBIAN_FRONTEND=noninteractive

# Essentials: developer tools, build tools, OpenBLAS
RUN apt-get update && apt-get install -y --no-install-recommends \
    apt-utils git curl vim unzip openssh-client wget \
    build-essential cmake \
    libopenblas-dev \
    libsm6 libxext6 libglib2.0-0

#
# Python 3.5
#
# For convenience, alias (but don't sym-link) python & pip to python3 & pip3 as recommended in:
# http://askubuntu.com/questions/351318/changing-symlink-python-to-python3-causes-problems
#
# Get version specific pip as pip as dropped support for Python 2 and 3.5
RUN apt-get install -y --no-install-recommends python3.5 python3.5-dev python3-pip python3-tk python3-wheel && \
    pip3 install --no-cache-dir --upgrade "pip < 9.0.4" "setuptools < 40.0.0" && \
    echo "alias python='python3'" >> /root/.bash_aliases && \
    echo "alias pip='pip3'" >> /root/.bash_aliases
# Pillow and it's dependencies
RUN apt-get install -y --no-install-recommends libjpeg-dev zlib1g-dev
RUN pip3 install --no-cache-dir Pillow
# Science libraries and other common packages
RUN pip3 install --no-cache-dir \
    wheel numpy==1.17.0 scipy scikit-learn==0.19 scikit-image pandas matplotlib Cython requests opencv-python==3.4.1.15

#
# Jupyter Notebook
#
# Allow access from outside the container, and skip trying to open a browser.
# NOTE: disable authentication token for convenience. DON'T DO THIS ON A PUBLIC SERVER.
RUN apt-get install -y --no-install-recommends libffi-dev
RUN pip3 install --no-cache-dir jupyter && \
    mkdir /root/.jupyter && \
    echo "c.NotebookApp.ip = '*'" \
         "\nc.NotebookApp.open_browser = False" \
         "\nc.NotebookApp.token = ''" \
         > /root/.jupyter/jupyter_notebook_config.py
EXPOSE 8888

#
# Tensorflow 1.6.0 - CPU
#
RUN pip3 install --no-cache-dir --upgrade tensorflow 

# Expose port for TensorBoard
EXPOSE 6006

# #
# # OpenCV 3.4.1
# #
# # Dependencies
# RUN apt-get install -y --no-install-recommends \
#     libjpeg8-dev libtiff5-dev libjasper-dev libpng12-dev \
#     libavcodec-dev libavformat-dev libswscale-dev libv4l-dev libgtk2.0-dev \
#     liblapacke-dev checkinstall
# # Get source from github
# RUN git clone -b 3.4.1 --depth 1 https://github.com/opencv/opencv.git /usr/local/src/opencv
# # Compile
# RUN cd /usr/local/src/opencv && mkdir build && cd build && \
#     cmake -D CMAKE_INSTALL_PREFIX=/usr/local \
#           -D BUILD_TESTS=OFF \
#           -D BUILD_PERF_TESTS=OFF \
#           -D PYTHON_DEFAULT_EXECUTABLE=$(which python3) \
#           .. && \
#     make -j"$(nproc)" && \
#     make install

# #
# # Caffe
# #
# # Dependencies
# RUN apt-get install -y --no-install-recommends \
#     cmake libprotobuf-dev libleveldb-dev libsnappy-dev libopencv-dev \
#     libhdf5-serial-dev protobuf-compiler liblmdb-dev libgoogle-glog-dev \
#     libboost-all-dev && \
#     pip3 install lmdb
# # Get source. Use master branch because the latest stable release (rc3) misses critical fixes.
# RUN git clone -b master --depth 1 https://github.com/BVLC/caffe.git /usr/local/src/caffe
# # Python dependencies
# RUN pip3 --no-cache-dir install -r /usr/local/src/caffe/python/requirements.txt
# # Compile
# RUN cd /usr/local/src/caffe && mkdir build && cd build && \
#     cmake -D CPU_ONLY=ON -D python_version=3 -D BLAS=open -D USE_OPENCV=ON .. && \
#     make -j"$(nproc)" all && \
#     make install
# # Enivronment variables
# ENV PYTHONPATH=/usr/local/src/caffe/python:$PYTHONPATH \
# 	PATH=/usr/local/src/caffe/build/tools:$PATH
# # Fix: old version of python-dateutil breaks caffe. Update it.
# RUN pip3 install --no-cache-dir python-dateutil --upgrade

# #
# # Java
# #
# # Install JDK (Java Development Kit), which includes JRE (Java Runtime
# # Environment). Or, if you just want to run Java apps, you can install
# # JRE only using: apt install default-jre
# RUN apt-get install -y --no-install-recommends default-jdk

#
# Keras 2.1.5
#
RUN pip3 install --no-cache-dir --upgrade h5py pydot_ng keras==2.1.5

#
# PyTorch 0.3.1
#
RUN pip3 install http://download.pytorch.org/whl/cpu/torch-0.3.1-cp35-cp35m-linux_x86_64.whl && \
    pip3 install torchvision

#
# PyCocoTools
#
# Using a fork of the original that has a fix for Python 3.
# I submitted a PR to the original repo (https://github.com/cocodataset/cocoapi/pull/50)
# but it doesn't seem to be active anymore.
RUN pip3 install --no-cache-dir git+https://github.com/waleedka/coco.git#subdirectory=PythonAPI

# 
# Upgrade pip3 to latest version and install imgaug, downgrade jsonschema and jupyter-client
#
RUN pip3 install --upgrade pip
RUN pip3 install --no-cache-dir imgaug jsonschema==2.6.0 jupyter-client==6.1.12

WORKDIR "/host"
CMD ["/bin/bash"]