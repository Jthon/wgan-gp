# wgan-gp
An tensorflow implementation of Paper ["Improved Training of Wasserstein GANs"](https://arxiv.org/pdf/1704.00028.pdf).
## Resources
[mnist dataset](http://yann.lecun.com/exdb/mnist/)
[cifar10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html)
## Architecture
#### Generator
noise_input:dim=128
fc: noise_input->dim=4*4*256,relu
decnn1: fc->dim=8*8*128,ksize=5,strides=2,relu
decnn2: decnn1->dim=16*16*64,ksize=5,strides=2,relu
output: decnn2->dim=32*32*1(3 for cifar),ksize=5,strides=2,tanh
#### Critic
image_input:dim=32*32*1(3 for cifar)
cnn1:image_input->16*16*64,ksize=5,strides=2,lrelu
cnn2:cnn1->dim=8*8*128,ksize=5,strides=2,lrelu
cnn3:cnn2->dim=4*4*256,ksize=5,strides=2,lrelu
output:cnn3->fc,dim=1
## Hyperparameters
batch_size=32
lr_rate=1e-4
optimizer=Adam
gp_lambda=10
critic_iter=5
## Result
   |epoch=1|epoch=20|g_loss|d_loss
---|:--:|---:|---:
mnist|![mnist_epoch=1](https://github.com/Jthon/wgan-gp/blob/master/result/mnist/epoch%3D1.jpg)|![mnist_epoch=20](https://github.com/Jthon/wgan-gp/blob/master/result/mnist/epoch%3D20.jpg)|内容|内容|
cifar|![cifar_epoch=1](https://github.com/Jthon/wgan-gp/blob/master/result/cifar/epoch%3D1.jpg)|![cifar_epoch=20](https://github.com/Jthon/wgan-gp/blob/master/result/cifar/epoch%3D20.jpg)|内容|内容|

