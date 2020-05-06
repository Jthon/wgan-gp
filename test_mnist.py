import load_mnist
import config as cfg
import numpy as np
import cv2
mnist_dataset=load_mnist.MNIST(cfg.params["mnist_image"],cfg.params["mnist_label"])
for i in range(0,mnist_dataset.datanum):
    signal=False
    print("num=%d"%int(mnist_dataset.labels[i]))
    while True:
        key=cv2.waitKey(5)
        if key==13:
            break
        if key==27:
            signal=True
            break
        cv2.imshow("image",np.array(mnist_dataset.images[i],np.uint8))
    if signal==True:
        break
