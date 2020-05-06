import pickle
import config as cfg
import os
import numpy as np

class CIFAR10:
    def __init__(self,data_dir):
        self.num_vis,self.label_names=self.Load_prefix(data_dir+"batches.meta")
        self.image_size=int(np.math.sqrt(self.num_vis/3))
        self.images=np.empty(shape=(0,self.image_size,self.image_size,3),dtype=np.uint8)
        self.labels=np.empty(shape=(0),dtype=np.int32)
        self.LoadImage_Label(data_dir)
        assert self.images.shape[0]==self.labels.shape[0]
        self.datanum=self.images.shape[0]
    def Load_prefix(self,prefix_path):
        with open(prefix_path, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
            num_vis=dict[b'num_vis']
            label_names=str(dict[b'label_names']).replace('b','')
            label_names=label_names.replace('[','').replace(']','').split(",")
        return int(num_vis),label_names
    def LoadImage_Label(self,data_dir):
        candidate_file=[]
        for filename in os.listdir(data_dir):
            if "batch" in filename and "meta" not in filename:
                candidate_file.append(filename)
        for filename in candidate_file:
            with open(data_dir+filename, 'rb') as fo:
                dict = pickle.load(fo, encoding='bytes')
                red_channel=np.reshape(dict[b"data"][:,0:self.image_size*self.image_size],(dict[b"data"].shape[0],self.image_size,self.image_size,1))
                green_channel=np.reshape(dict[b"data"][:,self.image_size*self.image_size:2*self.image_size*self.image_size],(dict[b"data"].shape[0],self.image_size,self.image_size,1))
                blue_channel=np.reshape(dict[b"data"][:,2*self.image_size*self.image_size:3*self.image_size*self.image_size],(dict[b"data"].shape[0],self.image_size,self.image_size,1))
                image_mat=np.concatenate((blue_channel,green_channel,red_channel),axis=3)
                label_list=np.array(dict[b'labels'],np.int32)
                self.images=np.concatenate((self.images,image_mat),axis=0)
                self.labels=np.concatenate((self.labels,label_list),axis=0)
    def random_batch(self,batch_size):
        index=np.random.permutation(np.arange(0,self.datanum,1))[0:batch_size]
        batch_images=self.images[index]
        batch_labels=self.labels[index]
        return batch_images,batch_labels

    
    