import struct
import numpy as np
import os
import config as cfg
import cv2
class MNIST:
    def __init__(self,ubyte_image_path,ubyte_label_path):
        self.images=self.LoadImage(ubyte_image_path)
        self.labels=self.LoadLabel(ubyte_label_path)
        assert self.images.shape[0]==self.labels.shape[0]
        self.datanum=self.images.shape[0]
    def LoadImage(self,ubyte_image_path):
        binfile=open(ubyte_image_path,'rb')
        buffers=binfile.read()
        head=struct.unpack_from('>IIII', buffers,0)
        offset=struct.calcsize('>IIII')
        imgNum=head[1]
        width=head[2]
        height=head[3]
        bits=imgNum*width*height
        bitsString='>'+str(bits)+'B'
        imgs=struct.unpack_from(bitsString,buffers,offset)
        binfile.close()
        imgs=np.reshape(imgs, [imgNum,width,height])
        return imgs
    def LoadLabel(self,ubyte_label_path):
        binfile=open(ubyte_label_path,'rb')
        buffers=binfile.read()
        head=struct.unpack_from('>II',buffers,0)
        labelNum=head[1]
        offset=struct.calcsize('>II')
        numString='>'+str(labelNum)+"B"
        labels=struct.unpack_from(numString,buffers,offset)
        binfile.close()
        labels=np.reshape(labels,[labelNum,1])
        return labels
    def random_batch(self,batch_size):
        index=np.random.permutation(np.arange(0,self.datanum,1))[0:batch_size]
        batch_image=self.images[index]
        batch_label=self.labels[index]
        batch_image=np.reshape(batch_image,(-1,28,28,1))
        return batch_image,batch_label

