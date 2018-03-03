#coding=utf-8

from utils import load_data
import numpy as np
from network_model import unet_3d_model
import matplotlib.pyplot as plt
import random
import os
#------------------ global settings ------------------#
REL_FILE_PATH = './plutdata'
SAVEPSNR = './savepsnr'
ind1 = random.randint(0,99)
ind2 = random.randint(0,15)
start_point = [ind1,ind1,ind2]
end_point = [876,900,100]
stride = [80,80,10]
max_epochs = 500
batch_size = 20
TRAINDATA_SAVE_PATH = './traindata_save'
if not os.path.exists(TRAINDATA_SAVE_PATH):
    os.mkdir(TRAINDATA_SAVE_PATH)
mode = 'train'
if mode == 'train':
    patch_size = [10, 10, 10]
elif mode == 'test':
    patch_size = [256, 256, 256]
#------------------ global settings ------------------#

CNNclass = unet_3d_model(batch_size=batch_size,
                                 input_size=patch_size,
                                 kernel_size=[3,3,3],
                                 in_channel=1,
                                 num_filter=16,
                                 stride=[1,1,1],
                                 epochs=2)
model = CNNclass.build_model()


if mode == 'train':
    for epoch in range(max_epochs):
        data_label_epoch, data_epoch, _ = load_data(rel_file_path=REL_FILE_PATH,
                                                    start_point=start_point,
                                                    end_point=end_point,
                                                    patch_size=patch_size,
                                                    stride=stride,
                                                    traindata_save=TRAINDATA_SAVE_PATH)
        data_epoch = np.expand_dims(data_epoch, axis=4)
        data_label_epoch = np.expand_dims(data_label_epoch, axis=4)

        train_data_size = np.shape(data_epoch)[0]  # 46656


        print "-------------------epoch:{}------------------".format(epoch+1)
        real_epoch = 0
        if (epoch+1)%2 == 0:
            real_epoch = 1

        model, hist = CNNclass.train_model(model=model,
                                          train_data=data_epoch,
                                          train_label=data_label_epoch,
                                          real_epochs=real_epoch)

elif mode == 'test':
    _, _, test_data = load_data(rel_file_path=REL_FILE_PATH,
                                start_point=start_point,
                                end_point=end_point,
                                patch_size=patch_size,
                                stride=stride,
                                traindata_save=TRAINDATA_SAVE_PATH)

    onedata = np.concatenate((test_data[0, :, :, :], test_data[1, :, :, :]), axis=2)  # 876*900*160
    onedata_test = onedata[:, :, :patch_size[2]]

    # normalize to [0,1]
    max_train_temp = np.max(onedata_test)
    min_train_temp = np.min(onedata_test)
    onedata_test = (onedata_test - min_train_temp) / (max_train_temp - min_train_temp)

    std_train_temp = np.mean(onedata_test)

    noise_level = random.randint(1, 10) * 1e-2
    onedata_test_noise = np.random.normal(0, noise_level * std_train_temp, onedata_test.shape) + onedata_test

    denoised = CNNclass.test_model(model=model,test_data=onedata_test_noise)

    i = 0
    for j in range(0,np.shape(denoised)[1],40):
        temp = denoised[i,:,:,:]
        temp2 = onedata_test_noise[i,:,:,:,0]
        temp3 = onedata_test[i,:,:,:,0]
        plt.figure(3*(j-1))
        plt.imshow(temp[:,j,:].reshape(256, 256), cmap='gray')
        plt.figure(3*(j-1)+1)
        plt.imshow(temp2[:, j , :].reshape(256, 256), cmap='gray')
        plt.figure(3*(j-1)+2)
        plt.imshow(temp3[:, j, :].reshape(256, 256), cmap='gray')
        plt.show()







