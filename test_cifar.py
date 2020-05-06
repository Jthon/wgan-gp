import load_cifar
import config as cfg
import cv2
import numpy as np
cifar10_dataset=load_cifar.CIFAR10(cfg.params["cifar_data"])
for i in range(0,cifar10_dataset.images.shape[0]):
    signal=False
    print("class="+cifar10_dataset.label_names[cifar10_dataset.labels[i]])
    while True:
        key=cv2.waitKey(5)
        if key==27:
            signal=True
            break
        elif key==13:
            break
        cv2.imshow("image",cv2.resize(np.array(cifar10_dataset.images[i],np.uint8),(64,64)))
    if signal==True:
        break