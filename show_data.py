# -*- coding: utf-8 -*-
import scipy.io as sio
from PIL import Image
import scipy
import numpy as np
import math

def kernel_visuzlize(model,path,epoch):
    model.load_weights('model_weights.h5')
    layers = model.layers
    len_layer = len(layers)
    for i in range(len_layer):
        layer = layers[i]
        if 'conv3d' in layer.name:
            weights = layer.get_weights()
            kernel = weights[0]
            bias = weights[1]
            kernel_all = np.zeros([np.shape(kernel)[0]*np.shape(kernel)[4],np.shape(kernel)[1]*np.shape(kernel)[3],3])
            for j in range(np.shape(kernel)[4]):
                for jj in range(np.shape(kernel)[3]):
                    temp = np.squeeze(kernel[:,:,:3,jj,j])
                    temp = (temp - np.min(temp)) / (np.max(temp) - np.min(temp))
                    kernel_all[j*np.shape(kernel)[0]:(j+1)*np.shape(kernel)[1],jj*(np.shape(kernel)[1]):(jj+1)*(np.shape(kernel)[1]),:]\
                        =temp
            scale = 5
            kernel_show = np.zeros([np.shape(kernel_all)[0] * scale, np.shape(kernel_all)[1] * scale,np.shape(kernel_all)[2]])
            for ii in range(np.shape(kernel_all)[0]):
                for jj in range(np.shape(kernel_all)[1]):
                    kernel_show[ii * scale:(ii + 1) * scale, jj * scale:(jj + 1) * scale,:] = \
                        np.repeat(np.repeat(np.reshape(kernel_all[ii, jj,:],[1,1,np.shape(kernel_all)[2]]),scale,axis=0),scale,axis=1)
            scipy.misc.imsave(path + '/'+layer.name+'_%d_epoch.png' % epoch, kernel_show)
        elif 'batch_normalization' in layer.name:
            weights = layer.get_weights()
            beta_gamma = np.zeros([len(weights),np.shape(weights[0])[0]])
            for i in range(len(weights)):
                temp = weights[i]
                temp = (temp - np.min(temp)) / (np.max(temp) - np.min(temp))
                beta_gamma[i,:] = temp
            scale = 5
            beta_gamma_show = np.zeros(
                [np.shape(beta_gamma)[0] * scale, np.shape(beta_gamma)[1] * scale])
            for ii in range(np.shape(beta_gamma)[0]):
                for jj in range(np.shape(beta_gamma)[1]):
                    beta_gamma_show[ii * scale:(ii + 1) * scale, jj * scale:(jj + 1) * scale] = \
                        beta_gamma[ii, jj]
            scipy.misc.imsave(path + '/' + layer.name + '_%d_epoch.png' % epoch, beta_gamma_show)
