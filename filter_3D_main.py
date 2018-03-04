#coding=utf-8

from utils import load_data
import numpy as np
from network_model import unet_3d_model
import matplotlib.pyplot as plt
import random
import os
import scipy
from keras.utils import plot_model
from show_data import kernel_visuzlize
#------------------ global settings ------------------#
REL_FILE_PATH = './plutdata'
SAVEPSNR = './savepsnr'
ind1 = random.randint(0,99)
ind2 = random.randint(0,15)
start_point = [ind1,ind1,ind2]
end_point = [876,900,100]
stride = [80,80,10]
max_epochs = 1000
batch_size = 20
TRAINDATA_SAVE_PATH = './traindata_save'
if not os.path.exists(TRAINDATA_SAVE_PATH):
    os.mkdir(TRAINDATA_SAVE_PATH)
TEST_RESULT_SAVE_PATH = './test_result'
if not os.path.exists(TEST_RESULT_SAVE_PATH):
    os.mkdir(TEST_RESULT_SAVE_PATH)
kernel_path = './kernel_save'
if not os.path.exists(kernel_path):
    os.mkdir(kernel_path)
#train/test/showkernel/model_visualize/kernel_visualize
mode = 'train'
if mode == 'train':
    patch_size = [40, 40, 40]
else:
    patch_size = [876, 900, 4]
    batch_size = 1
#------------------ global settings ------------------#

CNNclass = unet_3d_model(batch_size=batch_size,
                                 input_size=patch_size,
                                 kernel_size=[3,3,3],
                                 in_channel=1,
                                 num_filter=16,
                                 stride=[1,1,1],
                                 epochs=5)
model = CNNclass.build_model()


if mode == 'train':
    for epoch in range(max_epochs):
        ind1 = random.randint(0, 99)
        ind2 = random.randint(0, 15)
        start_point = [ind1, ind1, ind2]
        data_label_epoch, data_epoch, _ = load_data(rel_file_path=REL_FILE_PATH,
                                                    start_point=start_point,
                                                    end_point=end_point,
                                                    patch_size=patch_size,
                                                    stride=stride,
                                                    traindata_save=TRAINDATA_SAVE_PATH)
        data_epoch = np.expand_dims(data_epoch, axis=4)
        data_label_epoch = np.expand_dims(data_label_epoch, axis=4)

        train_data_size = np.shape(data_epoch)[0]  # 46656
        if epoch == 0:
            print '[*] kernel visualize'
            kernel_visuzlize(model, kernel_path, epoch)

        if (epoch + 1) % 1 == 0:
            print '[*] train result visualize'
            ind = np.arange(np.shape(data_epoch)[0])
            ind = np.random.permutation(ind)
            data_test = data_epoch[ind[:2], :, :, :, :]
            data_label_test = data_label_epoch[ind[:2], :, :, :, :]

            denoise_test = CNNclass.test_model(model=model, test_data=data_test)
            for j in range(np.shape(data_test)[0]):
                for i in range(8):
                    if i % 3 == 0:
                        indd = random.randint(0, np.shape(data_test)[3] - 1)
                        temp1 = denoise_test[j, :, :, indd, 0]
                        temp2 = data_test[j, :, :, indd, 0]
                        temp3 = data_label_test[j, :, :, indd, 0]
                    elif i % 2 == 0:
                        indd = random.randint(0, np.shape(data_test)[2] - 1)
                        temp1 = denoise_test[j, :, indd, :, 0]
                        temp2 = data_test[j, :, indd, :, 0]
                        temp3 = data_label_test[j, :, indd, :, 0]
                    else:
                        indd = random.randint(0, np.shape(data_test)[1] - 1)
                        temp1 = denoise_test[j, indd, :, :, 0]
                        temp2 = data_test[j, indd, :, :, 0]
                        temp3 = data_label_test[j, indd, :, :, 0]
                    if i == 0:
                        result = np.concatenate((temp1.squeeze(), temp2.squeeze(), temp3.squeeze()),
                                                axis=1)
                    else:
                        temp = np.concatenate((temp1.squeeze(), temp2.squeeze(), temp3.squeeze()),
                                              axis=1)
                        result = np.concatenate((result, temp), axis=0)
            scipy.misc.imsave('./train_result' + '/denoise_noisedata_label%d.png' % epoch, result)

        print "-------------------epoch:{}------------------".format(epoch+1)
        real_epoch = 0
        if (epoch+1)%2 == 0:
            real_epoch = 1

        model, hist = CNNclass.train_model(model=model,
                                          train_data=data_epoch,
                                          train_label=data_label_epoch,
                                          real_epochs=real_epoch)
        if real_epoch == 1:
            print '[*] kernel visualize'
            kernel_visuzlize(model, kernel_path, epoch)


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

    noise_level = random.randint(15, 20) * 1e-2
    onedata_test_noise = np.random.normal(0, noise_level * std_train_temp, onedata_test.shape) + onedata_test

    onedata_test_noise = np.reshape(onedata_test_noise,[1,np.shape(onedata_test_noise)[0],np.shape(onedata_test_noise)[1],np.shape(onedata_test_noise)[2],1])
    denoised = CNNclass.test_model(model=model, test_data=onedata_test_noise)

    for i in range(patch_size[2]):
        scipy.misc.imsave(TEST_RESULT_SAVE_PATH + '/%d_%.2flabel.png' % (i, noise_level), onedata_test[:, :, i])
        scipy.misc.imsave(TEST_RESULT_SAVE_PATH + '/%d_%.2fmdenoise.png' % (i, noise_level),
                          np.squeeze(denoised[:, :, :, i, 0]))
        scipy.misc.imsave(TEST_RESULT_SAVE_PATH + '/%d_%.2fnoisedata.png' % (i, noise_level),
                          np.squeeze(onedata_test_noise[:, :, :, i, 0]))
    '''
    onedata_test_extract = []
    for i in range(0, np.shape(onedata_test)[0] - patch_size[0] + 1, patch_size[0]):
        for j in range(0, np.shape(onedata_test)[1] - patch_size[1] + 1, patch_size[1]):
            for k in range(0, np.shape(onedata_test)[2] - patch_size[2] + 1, patch_size[2]):
                temp_noise = onedata_test_noise[i:i + patch_size[0], j:j + patch_size[1], k:k + patch_size[2]]
                onedata_test_extract.append(temp_noise)
    if np.shape(onedata_test)[0] % patch_size[0] != 0:
        for j in range(0, np.shape(onedata_test)[1] - patch_size[1] + 1, patch_size[1]):
            for k in range(0, np.shape(onedata_test)[2] - patch_size[2] + 1, patch_size[2]):
                temp_noise = onedata_test_noise[np.shape(onedata_test)[0] - patch_size[0]:, j:j + patch_size[1],
                             k:k + patch_size[2]]
                onedata_test_extract.append(temp_noise)
    if np.shape(onedata_test)[1] % patch_size[1] != 0:
        for i in range(0, np.shape(onedata_test)[0] - patch_size[0] + 1, patch_size[0]):
            for k in range(0, np.shape(onedata_test)[2] - patch_size[2] + 1, patch_size[2]):
                temp_noise = onedata_test_noise[i:i + patch_size[0], np.shape(onedata_test)[1] - patch_size[1]:,
                             k:k + patch_size[2]]
                onedata_test_extract.append(temp_noise)
    if np.shape(onedata_test)[0] % patch_size[0] != 0 and np.shape(onedata_test)[1] % patch_size[1] != 0:
        for k in range(0, np.shape(onedata_test)[2] - patch_size[2] + 1, patch_size[2]):
            temp_noise = onedata_test_noise[np.shape(onedata_test)[0] - patch_size[0]:,
                         np.shape(onedata_test)[1] - patch_size[1]::,
                         k:k + patch_size[2]]
            onedata_test_extract.append(temp_noise)

    onedata_test_extract = np.expand_dims(onedata_test_extract, axis=4)
    denoise = np.zeros(np.shape(onedata_test_extract))
    for i in range(np.shape(onedata_test_extract)[0] // batch_size):
        denoise[i * batch_size:(i + 1) * batch_size, :, :, :, :] = \
            CNNclass.test_model(model=model,
                                test_data=onedata_test_extract[i * batch_size:(i + 1) * batch_size, :, :, :, :])
    if np.shape(onedata_test_extract)[0] % batch_size != 0:
        denoise[np.shape(onedata_test_extract)[0] // batch_size * batch_size:, :, :, :, :] = \
            CNNclass.test_model(model=model,
                                test_data=onedata_test_extract[np.shape(onedata_test)[0] // batch_size * batch_size:, :, :, :, :])
    #denoised = CNNclass.test_model(model=model,test_data=onedata_test_noise)

    count = 0
    denoise_onedata = np.zeros(np.shape(onedata_test))
    for i in range(0, np.shape(onedata_test)[0] - patch_size[0] + 1, patch_size[0]):
        for j in range(0, np.shape(onedata_test)[1] - patch_size[1] + 1, patch_size[1]):
            denoise_onedata[i:i + patch_size[0], j:j + patch_size[1], k:k + patch_size[2]] \
                = denoise[count, :, :, :, 0]
            count = count + 1
    if np.shape(onedata_test)[0] % patch_size[0] != 0:
        for j in range(0, np.shape(onedata_test)[1] - patch_size[1] + 1, patch_size[1]):
            for k in range(0, np.shape(onedata_test)[2] - patch_size[2] + 1, patch_size[2]):
                ind = np.shape(onedata_test)[0] - np.shape(onedata_test)[0] // patch_size[0] * patch_size[0]
                denoise_onedata[-ind:, j:j + patch_size[1], k:k + patch_size[2]] \
                    = denoise[count, -ind:, :, :, 0]
                count = count + 1
    if np.shape(onedata_test)[1] % patch_size[1] != 0:
        for i in range(0, np.shape(onedata_test)[0] - patch_size[0] + 1, patch_size[0]):
            for k in range(0, np.shape(onedata_test)[2] - patch_size[2] + 1, patch_size[2]):
                ind = np.shape(onedata_test)[1] - np.shape(onedata_test)[1] // patch_size[1] * patch_size[1]
                denoise_onedata[i:i + patch_size[0], -ind:, k:k + patch_size[2]] \
                    = denoise[count, :, -ind:, :, 0]
                count = count + 1
    if np.shape(onedata_test)[0] % patch_size[0] != 0 and np.shape(onedata_test)[1] % patch_size[1] != 0:
        for k in range(0, np.shape(onedata_test)[2] - patch_size[2] + 1, patch_size[2]):
            ind1 = np.shape(onedata_test)[0] - np.shape(onedata_test)[0] // patch_size[0] * patch_size[0]
            ind2 = np.shape(onedata_test)[1] - np.shape(onedata_test)[1] // patch_size[1] * patch_size[1]
            denoise_onedata[-ind1:, -ind2:, :] = \
                denoise[count, -ind1:, -ind2:, :, 0]
            count = count + 1
    for i in range(np.shape(onedata_test)[2]):
        scipy.misc.imsave(TEST_RESULT_SAVE_PATH + '/%dlabel.png'%i, onedata_test[:,:,i])
        scipy.misc.imsave(TEST_RESULT_SAVE_PATH + '/%dmdenoised.png'%i, denoise_onedata[:,:,i])
        scipy.misc.imsave(TEST_RESULT_SAVE_PATH + '/%dnoisedata.png'%i, onedata_test_noise[:, :, i])
    '''
    print 'ok'

elif mode == 'model_visualize':
    plot_model(model, to_file='model.png')
elif mode == 'kernel_visualize':
    kernel_visuzlize(model,kernel_path,1)






