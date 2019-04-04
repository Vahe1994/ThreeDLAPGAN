# 3D Laplacian GAN for Point Cloud

## Introduction

The paper you can find here https://www.hse.ru/en/edu/vkr/219584552 (in. Russian).

This repository is devoted to the topic of three-dimensional point clouds generation. It is proposed to implement the architecture of a multilevel generative adversarial neural network. This work is based on the concept set forth in the article «Deep Generative Image Models using a Laplacian Pyramid of Adversarial Networks», whose authors propose the structure of a multilevel generative adversarial neural network, for obtaining a better quality in high resolution images, and on the article «Representation Learning and Adversarial Generation of 3D Point Clouds», showing the efficiency of point cloud generation by using the auto-encoder and generative adversarial network, which generates a latent representation, instead of direct generation of a point cloud. You should check he's repository https://github.com/optas/latent_3d_points ,because this one based on it. Code that provided from optas is based on tensorflow 1.3, and my code on Pytorch 0.3 +


## Dependencies

- Python 2.7+ with Numpy, Scipy and Matplotlib
- Pytorch 0.3 +
- Tensorflow (version 1.3)
- TFLearn

## Usage
Take a look at usage here https://github.com/optas/latent_3d_points . After you come back just take a look at new files in folders notebooks/ and src/

### Building the docker image

Build the docker container:
```bash
docker build -f Dockerfile --tag artonson/3dlapgan:latest .
```

Then enter the container by running it (don't forget to name the containers accordingly and remove them post-usage!):
```bash
docker run --rm -it --name 3ddl.artonson.0.3dlapgan --runtime=nvidia -v /home/artonson/repos/ThreeDLAPGAN:/code -p 3340:3340 artonson/3dlapgan:latest
```
Remember our container naming conventions: `3ddl.<username>.<gpu-ids-list>.<customsuffix>`.

After entering the container shell, you will be able to run the Jupyter notebook:
```bash
cd /code
jupyter notebook --NotebookApp.token=abcd --ip=0.0.0.0 --port 3340 --allow-root
```
and your token will be `abcd`.



## Examples of generated models for each level

First level(512)
1. https://plot.ly/~Vahe1994/274 2.https://plot.ly/~Vahe1994/264

Second level(1024)
1. https://plot.ly/~Vahe1994/276 2. https://plot.ly/~Vahe1994/266

Third level(2048)
1. https://plot.ly/~Vahe1994/278 2. https://plot.ly/~Vahe1994/268



