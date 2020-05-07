import load_mnist
import load_cifar
import config as cfg
import wgan_net 
import tensorflow as tf
import numpy as np
import cv2
import os
os.environ["CUDA_VISIBLE_DEVICES"] = cfg.params["which_gpu"]
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
if cfg.params["dataset"]=="mnist":
    data=load_mnist.MNIST(cfg.params["mnist_image"],cfg.params["mnist_label"])
elif cfg.params["dataset"]=="cifar":
    data=load_cifar.CIFAR10(cfg.params["cifar_data"])
    
offset_epoch=0
current_epoch=0
current_steps=0
log_steps=0
instance_num=data.datanum
writer=tf.summary.FileWriter(cfg.params["log_dir"]+cfg.params["dataset"])
latest_checkpoint=tf.train.latest_checkpoint(cfg.params["model_dir"])
assert (latest_checkpoint!=None)==(cfg.params["restore"])
net=wgan_net.Wgan_network()
saver=tf.train.Saver(max_to_keep=1)
with tf.Session() as sess:
    if cfg.params["restore"]==True:
        saver.restore(sess,latest_checkpoint)
        offset_epoch=int(latest_checkpoint.spilt("=")[1].split(".")[0])
    else:
        sess.run(tf.global_variables_initializer())

    while current_epoch<cfg.params["epoch"]:
        for i in range(0,cfg.params["critic_iter"]):
            image_batch,_=data.random_batch(cfg.params["batch_size"])
            image_batch=np.array(image_batch/127.5-1,np.float32)
            noise_batch=np.random.uniform(-1., 1.,size=[cfg.params["batch_size"],128])
            d_loss,_=sess.run([net.d_loss,net.d_train_op],feed_dict={net.image_input:image_batch,net.noise_input:noise_batch})
        
        noise_batch=np.random.uniform(-1., 1.,size=[cfg.params["batch_size"],128])
        fake,g_loss,_=sess.run([net.fake,net.g_loss,net.g_train_op],feed_dict={net.noise_input:noise_batch})
        print("d_loss=%f,g_loss=%f"%(d_loss,g_loss))
        image_summary,loss_summary=sess.run([net.image_summary,net.loss_summary],feed_dict={net.image_input:image_batch,net.noise_input:noise_batch})
        writer.add_summary(loss_summary,global_step=log_steps)
        
        current_steps+=1
        log_steps+=1
        if current_steps*cfg.params["batch_size"]>=instance_num:
            current_epoch+=1
            current_steps=0
            saver.save(sess,cfg.params["model_dir"]+cfg.params["dataset"]+"/epoch="+str(current_epoch+offset_epoch)+".ckpt")
            writer.add_summary(image_summary,global_step=current_epoch)

            #write current epoch sample result
            if cfg.params['dataset']=="mnist":
                empty_image=np.zeros(shape=(5*32,5*32,1),dtype=np.uint8)
            elif cfg.params['dataset']=="cifar":
                empty_image=np.zeros(shape=(5*32,5*32,3),dtype=np.uint8)
            for i in range(0,5):
                for j in range(0,5):
                    empty_image[i*32:(i+1)*32,j*32:(j+1)*32]=np.array((fake[5*i+j]+1)*127.5,np.uint8)
            cv2.imwrite(cfg.params["result_dir"]+cfg.params["dataset"]+"/epoch="+str(current_epoch+offset_epoch)+".jpg",empty_image)

        
        cv2.imshow("test",np.array((fake[0]+1)*127.5,np.uint8))
        cv2.waitKey(5)


   

    




