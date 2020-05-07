# wgan-gp
An tensorflow implementation of Paper ["Improved Training of Wasserstein GANs"](https://arxiv.org/pdf/1704.00028.pdf).
## Resources
[mnist dataset](http://yann.lecun.com/exdb/mnist/) <br>
[cifar10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html) <br>
## Architecture
#### Generator
**noise_input**:dim=128 <br>
**fc**: noise_input->dim=4x4x256,relu <br>
**decnn1**: fc->dim=8x8x128,ksize=5,strides=2,relu <br>
**decnn2**: decnn1->dim=16x16x64,ksize=5,strides=2,relu <br>
**output**: decnn2->dim=32x32x1(3 for cifar),ksize=5,strides=2,tanh <br>
#### Critic
**image_input**:dim=32x32x1(3 for cifar) <br>
**cnn1**:image_input->16x16x64,ksize=5,strides=2,lrelu <br>
**cnn2**:cnn1->dim=8x8x128,ksize=5,strides=2,lrelu <br>
**cnn3**:cnn2->dim=4x4x256,ksize=5,strides=2,lrelu <br>
**output**:cnn3->fc,dim=1 <br>
## Hyperparameters
**batch_size**=32 <br>
**lr_rate**=1e-4  <br>
**optimizer**=Adam <br>
**gp_lambda**=10 <br>
**critic_iter**=5 <br>
## Result
||epoch=1|epoch=20|g_loss|d_loss
---|:--:|:--:|:--:|:--:|
mnist|![mnist_epoch=1](https://github.com/Jthon/wgan-gp/blob/master/result/mnist/epoch%3D1.jpg)|![mnist_epoch=20](https://github.com/Jthon/wgan-gp/blob/master/result/mnist/epoch%3D20.jpg)|![mnist_g_loss](https://github.com/Jthon/wgan-gp/blob/master/result/mnist_g.png)|![mnist_d_loss](https://github.com/Jthon/wgan-gp/blob/master/result/mnist_d.png)|
cifar|![cifar_epoch=1](https://github.com/Jthon/wgan-gp/blob/master/result/cifar/epoch%3D1.jpg)|![cifar_epoch=20](https://github.com/Jthon/wgan-gp/blob/master/result/cifar/epoch%3D20.jpg)|![cifar_g_loss](https://github.com/Jthon/wgan-gp/blob/master/result/cifar_g.png)|![cifar_d_loss](https://github.com/Jthon/wgan-gp/blob/master/result/cifar_d.png)|

