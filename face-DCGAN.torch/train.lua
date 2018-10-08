--[[ Deep Convolutional Generative Adversarial Network (DCGAN).

Using deep convolutional generative adversarial networks (DCGAN) 
to generate face images from a noise distribution.

References:
    - Generative Adversarial Nets. Goodfellow et al. arXiv: 1406.2661.
    - Unsupervised Representation Learning with Deep Convolutional 
    Generative Adversarial Networks. A Radford, L Metz, S Chintala. 
    arXiv: 1511.06434.

Links:
    - [GAN Paper](https://arxiv.org/pdf/1406.2661.pdf).
    - [DCGAN Paper](https://arxiv.org/abs/1511.06434.pdf).

Author: Prabhsimran Singh
Project: https://github.com/pskrunner14/face-DCGAN/tree/master/face-DCGAN.torch ]]--

require 'nn'
require 'torch'
require 'optim'