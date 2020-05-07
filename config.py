import argparse
import os
parser = argparse.ArgumentParser()
parser.add_argument("--dataset",type=str,default="cifar")
parser.add_argument("--mnist_image",type=str,default="./dataset/mnist/train-images.idx3-ubyte")
parser.add_argument("--mnist_label",type=str,default="./dataset/mnist/train-labels.idx1-ubyte")

parser.add_argument("--cifar_data",type=str,default="./dataset/cifar10/")

parser.add_argument("--result_dir",type=str,default="./result/")
parser.add_argument("--model_dir",type=str,default="./model/")
parser.add_argument("--log_dir",type=str,default="./log/")

parser.add_argument("--restore",type=bool,default=False)

parser.add_argument("--batch_size",type=int,default=32)
parser.add_argument("--lr_rate",type=float,default=1e-4)
parser.add_argument("--epoch",type=int,default=20)
parser.add_argument("--LAMBDA",type=int,default=10)
parser.add_argument('--critic_iter',type=int, default=5)
parser.add_argument('--which_gpu', type=str, default="1")

args = parser.parse_args()
# convert args to dictionary
params = vars(args)
if not os.path.exists(params["result_dir"]):
    os.mkdir(params["result_dir"])
if not os.path.exists(params["result_dir"]+params["dataset"]):
    os.mkdir(params["result_dir"]+params["dataset"])
if not os.path.exists(params["model_dir"]):
    os.mkdir(params["model_dir"])
if not os.path.exists(params["model_dir"]+params["dataset"]):
    os.mkdir(params["model_dir"]+params["dataset"])
if not os.path.exists(params["log_dir"]):
    os.mkdir(params["log_dir"])



