FROM tensorflow/tensorflow:1.3.0-devel-gpu-py3

LABEL maintainer="Alexey Artemov <artonson@yandex.ru>"

RUN pip3 --no-cache-dir install --upgrade \
        Pillow \
        h5py \
        ipykernel \
        jupyter \
        keras_applications \
        keras_preprocessing \
        matplotlib \
        mock \
        numpy \
        scipy \
        scikit-learn \
        scikit-image \
        pandas \
        Cython \
        nose \
        torch \
        torchvision \
        tflearn \
        && \
    python3 -m ipykernel.kernelspec

WORKDIR /root

# TensorBoard
EXPOSE 6006
# IPython
EXPOSE 8888

RUN pip3 uninstall -y tensorflow

