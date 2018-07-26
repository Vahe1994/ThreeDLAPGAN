# 3DLAPGAN

## Introduction

The paper you can find here https://www.hse.ru/en/edu/vkr/219584552(ln. russia).

This repository is devoted to the topic of three-dimensional point clouds generation. It is proposed to implement the architecture of a multilevel generative adversarial neural network. This work is based on the concept set forth in the article «Deep Generative Image Models using a Laplacian Pyramid of Adversarial Networks», whose authors propose the structure of a multilevel generative adversarial neural network, for obtaining a better quality in high resolution images, and on the article «Representation Learning and Adversarial Generation of 3D Point Clouds», showing the efficiency of point cloud generation by using the auto-encoder and generative adversarial network, which generates a latent representation, instead of direct generation of a point cloud. You should check he's repository https://github.com/optas/latent_3d_points ,because this one based on it. Code that provided from optas is based on tensorflow 1.3, and my code on Pytorch 0.3 +


## Dependencies

- Python 2.7+ with Numpy, Scipy and Matplotlib
- Pytorch 0.3 +
- Tensorflow (version 1.3)
- TFLearn

## Usage
Take a look at usage here https://github.com/optas/latent_3d_points . After you come back just take a look at new files in folders notebooks/ and src/

## Examples of generated models for each level

First level(512)
https://plot.ly/~Vahe1994/274

Second level(1024)
https://plot.ly/~Vahe1994/276

Third level(2048)
https://plot.ly/~Vahe1994/278



<iframe width="900" height="800" frameborder="0" scrolling="no" src="//plot.ly/~Vahe1994/278.embed"></iframe>

<div>
    <a href="https://plot.ly/~Vahe1994/278/?share_key=WM0FVbW7QlRqgagYQ6sqvD" target="_blank" title="track2048" style="display: block; text-align: center;"><img src="https://plot.ly/~Vahe1994/278.png?share_key=WM0FVbW7QlRqgagYQ6sqvD" alt="track2048" style="max-width: 100%;width: 600px;"  width="600" onerror="this.onerror=null;this.src='https://plot.ly/404.png';" /></a>
    <script data-plotly="Vahe1994:278" sharekey-plotly="WM0FVbW7QlRqgagYQ6sqvD" src="https://plot.ly/embed.js" async></script>
</div>
