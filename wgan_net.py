import tensorflow as tf
import tensorflow.layers as layers
import config as cfg
class Wgan_network:
    def __init__(self):
        self.noise_input=tf.placeholder(dtype=tf.float32,shape=[None,128])
        if cfg.params["dataset"]=="mnist":
            self.image_input=tf.placeholder(dtype=tf.float32,shape=[None,28,28,1])
            self.pad_image_input=tf.pad(self.image_input,[[0,0],[2,2],[2,2],[0,0]],constant_values=-1)
        elif cfg.params["dataset"]=="cifar":
            self.image_input=tf.placeholder(dtype=tf.float32,shape=[None,32,32,3])
            self.pad_image_input=self.image_input
        self.__BuildNet__()
        
        self.loss_summary,self.image_summary=self.__logger__(self.fake,self.d_loss,self.g_loss)

    def __discriminator__(self,image_input):
        with tf.variable_scope("discriminator",reuse=tf.AUTO_REUSE):
            cnn1=layers.conv2d(image_input,filters=64,kernel_size=5,strides=(2,2),padding="SAME")
            cnn1=tf.nn.leaky_relu(cnn1)
            cnn2=layers.conv2d(cnn1,filters=128,kernel_size=5,strides=(2,2),padding="SAME")
            cnn2=tf.nn.leaky_relu(cnn2)
            cnn3=layers.conv2d(cnn2,filters=256,kernel_size=5,strides=(2,2),padding="SAME")
            cnn3=tf.nn.leaky_relu(cnn3)
            logits=layers.dense(layers.flatten(cnn3),1)
            #print_Arch:
            print("Discriminator-Architecture")
            print("input:{}".format(image_input.shape))
            print("cnn1:{}".format(cnn1.shape))
            print("cnn2:{}".format(cnn2.shape))
            print("cnn3:{}".format(cnn3.shape))
            print("output:{}".format(logits.shape))
        return logits
    def __generator__(self,noise_input):
        with tf.variable_scope("generator"):
            noise_feature=layers.dense(noise_input,4*4*256)
            noise_feature=tf.nn.relu(noise_feature)
            noise_feature=tf.reshape(noise_feature,shape=(-1,4,4,256))

            decnn1=layers.conv2d_transpose(noise_feature,filters=128,kernel_size=5,strides=(2,2),padding="SAME")
            decnn1=tf.nn.relu(decnn1)

            decnn2=layers.conv2d_transpose(decnn1,filters=64,kernel_size=5,strides=(2,2),padding="SAME")
            decnn2=tf.nn.relu(decnn2)

            if cfg.params["dataset"]=="mnist":
                output=layers.conv2d_transpose(decnn2,filters=1,kernel_size=5,strides=(2,2),padding="SAME")
            elif cfg.params["dataset"]=="cifar":
                output=layers.conv2d_transpose(decnn2,filters=3,kernel_size=5,strides=(2,2),padding="SAME")
            output=tf.nn.tanh(output)
            #print_Arch:
            print("Generator-Architecture")
            print("input:{}".format(noise_input.shape))
            print("project:{}".format(noise_feature.shape))
            print("decnn1:{}".format(decnn1.shape))
            print("decnn2:{}".format(decnn2.shape))
            print("output:{}".format(output.shape))
            return output
    def __BuildNet__(self):

        self.fake=self.__generator__(self.noise_input)
        self.fake_logit=self.__discriminator__(self.fake)
        self.real_logit=self.__discriminator__(self.pad_image_input)

        self.wgan_d_loss = tf.reduce_mean(self.fake_logit)-tf.reduce_mean(self.real_logit)
        self.g_loss = -tf.reduce_mean(self.fake_logit)

        dis_param=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope="discriminator")
        gen_param=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope="generator")

        self.gp_loss = self.gradient_penalty()
        self.d_loss = self.wgan_d_loss + cfg.params["LAMBDA"] * self.gp_loss

        self.g_train_op = tf.train.AdamOptimizer(
            learning_rate=cfg.params["lr_rate"], beta1=0.5, beta2=0.9).minimize(self.g_loss, var_list=gen_param)
        self.d_train_op = tf.train.AdamOptimizer(
            learning_rate=cfg.params["lr_rate"], beta1=0.5, beta2=0.9).minimize(self.d_loss, var_list=dis_param)

    def gradient_penalty(self):
        epsilon=tf.random_uniform([cfg.params['batch_size'],1,1,1], 0.0, 1.0)
        x_hat=self.fake
        x_hat=epsilon*self.fake+(1-epsilon)*self.pad_image_input
        gp=tf.gradients(self.__discriminator__(x_hat),x_hat)[0]
        gp=tf.sqrt(tf.reduce_sum(tf.square(gp),axis=[1,2,3]))
        gp=tf.reduce_mean(tf.square(gp-1))
        return gp
    def __logger__(self,image,d_loss,g_loss):
        fake_summary=tf.summary.image("fake",image)
        g_summary=tf.summary.scalar("g_loss",g_loss)
        d_summary=tf.summary.scalar("d_loss",d_loss)

        loss_summary=tf.summary.merge([g_summary,d_summary])
        image_summary=fake_summary
        return loss_summary,image_summary

