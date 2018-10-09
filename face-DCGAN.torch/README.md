# face-DCGAN.torch

This is a Torch implementation of [https://arxiv.org/abs/1511.06434](https://arxiv.org/abs/1511.06434).

![DCGAN](../images/DCGAN.png)

## Usage

```
$ th train.lua --help

Trains a Deep Convolutional Generative Network (DCGAN)

See https://arxiv.org/abs/1511.06434.pdf for more details.

Optional arguments:
  -glr        learning rate for minimizing generator loss [0.0001]
  -g_beta1    value of `beta1` hyperparam for generator optimizer [0.5]
  -dlr        learning rate for minimizing discriminator loss [0.0001]
  -d_beta1    value of `beta1` hyperparam for discriminator optimizer [0.5]
  -batch_size mini-batch size for training the adversarial network [64]
  -num_epochs number of epochs to train the adversarial network [50]
  -gpu        if using GPU for training the adversarial network. Use 0 for CPU. [1]
```

## Built with

* LuaJIT
* Torch7

## References

* [Generative Adversarial Networks](https://arxiv.org/pdf/1406.2661)
* [Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks](https://arxiv.org/abs/1511.06434)