import tensorflow as tf
from keras.models import Sequential
from keras.layers.core import Activation
from keras.layers.convolutional import Conv3D
from keras.layers.normalization import BatchNormalization
from keras.callbacks import ModelCheckpoint

class unet_3d_model(object):
    def __init__(self,
                 batch_size=10,
                 input_size=[10,10,10],
                 kernel_size=[3,3,3],
                 in_channel=1,
                 num_filter = 16,
                 stride = [1,1,1],
                 epochs = 2):
        self.batch_size = batch_size
        self.input_size = input_size
        self.kernel_size = kernel_size
        self.in_channel = in_channel
        self.num_filter = num_filter
        self.stride = stride
        self.epochs = epochs
        self.build_model()

    def build_model(self):
        model = Sequential()
        # 1st layer
        model.add(Conv3D(self.num_filter,
                         self.kernel_size,
                         strides=1,
                         padding='same',
                         data_format='channels_last',
                         input_shape=(self.input_size[0],
                                      self.input_size[1],
                                      self.input_size[2],
                                      self.in_channel)))

        model.add(Activation('relu'))
        # 2nd layer
        model.add(Conv3D(self.num_filter,
                         self.kernel_size,
                         strides=1,
                         padding='same',
                         data_format='channels_last'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        # 3rd layer
        model.add(Conv3D(self.num_filter,
                         self.kernel_size,
                         strides=1,
                         padding='same',
                         data_format='channels_last'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        # 4th layer
        model.add(Conv3D(self.num_filter,
                         self.kernel_size,
                         strides=1,
                         padding='same',
                         data_format='channels_last'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        # 5th layer
        model.add(Conv3D(self.num_filter,
                         self.kernel_size,
                         strides=1,
                         padding='same',
                         data_format='channels_last'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        # 6th layer
        model.add(Conv3D(self.num_filter,
                         self.kernel_size,
                         strides=1,
                         padding='same',
                         data_format='channels_last'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        # 7th layer
        model.add(Conv3D(self.num_filter,
                         self.kernel_size,
                         strides=1,
                         padding='same',
                         data_format='channels_last'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        # 8th layer
        model.add(Conv3D(self.num_filter,
                         self.kernel_size,
                         strides=1,
                         padding='same',
                         data_format='channels_last'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        # 9th layer
        model.add(Conv3D(self.num_filter,
                         self.kernel_size,
                         strides=1,
                         padding='same',
                         data_format='channels_last'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))

        # 10th layer
        model.add(Conv3D(self.num_filter,
                         self.kernel_size,
                         strides=1,
                         padding='same',
                         data_format='channels_last'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))

        # 11th layer
        model.add(Conv3D(self.num_filter,
                         self.kernel_size,
                         strides=1,
                         padding='same',
                         data_format='channels_last'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))

        # 12th layer
        model.add(Conv3D(self.in_channel,
                         self.kernel_size,
                         strides=1,
                         padding='same',
                         data_format='channels_last'))

        model.compile(loss=self.loss, optimizer='Adam', metrics=[self.snr,'mean_squared_error',self.l1_loss,self.tv_loss])
        return model
    def loss(self,y,y_true):
        L1_loss_forward = tf.reduce_mean(tf.abs(y - y_true))

        tvDiff_loss_forward = self.tv_loss(y,y_true)
        loss = L1_loss_forward + tvDiff_loss_forward

        return loss

    def l1_loss(self,y,y_true):
        L1_loss_forward = tf.reduce_mean(tf.abs(y - y_true))
        return L1_loss_forward

    def tv_loss(self,y,y_true):
        pixel_num = self.input_size[0] * self.input_size[1]
        tv_lambda = 2000000
        for i in range(self.input_size[2]):
            if i == 0:
                tvDiff_loss_forward = \
                    tf.reduce_mean(tf.image.total_variation(y[:, :, :, i, :])) / pixel_num * tv_lambda / 10000
            else:
                tvDiff_loss_forward = tvDiff_loss_forward + \
                                      tf.reduce_mean(
                                          tf.image.total_variation(y[:, :, :, i, :])) / pixel_num * 200 / 10000
        for i in range(self.input_size[1]):
            if i == 0:
                tvDiff_loss_forward = \
                    tf.reduce_mean(tf.image.total_variation(y[:, :, i, :, :])) / pixel_num * tv_lambda / 10000
            else:
                tvDiff_loss_forward = tvDiff_loss_forward + \
                                      tf.reduce_mean(
                                          tf.image.total_variation(y[:, :, i, :, :])) / pixel_num * 200 / 10000
        for i in range(self.input_size[0]):
            if i == 0:
                tvDiff_loss_forward = \
                    tf.reduce_mean(tf.image.total_variation(y[:, i, :, :, :])) / pixel_num * tv_lambda / 10000
            else:
                tvDiff_loss_forward = tvDiff_loss_forward + \
                                      tf.reduce_mean(tf.image.total_variation(
                                          y[:, i, :, :, :])) / pixel_num * tv_lambda / 10000

        tvDiff_loss_forward = tvDiff_loss_forward / self.input_size[2] / self.input_size[1] / self.input_size[0]
        return tvDiff_loss_forward

    def snr(self,y,y_true):
        tmp_snr = tf.reduce_sum(tf.square(tf.abs(y_true))) / tf.reduce_sum(tf.square(tf.abs(y_true - y)))
        out = 10.0 * tf.log(tmp_snr) / tf.log(10.0)
        return out

    def conv3d(self,x,w):
        return tf.nn.conv3d(x,w,strides=[1,1,1,1,1],padding="SAME")

    def maxpool3d(self,x):
        return tf.nn.max_pool3d(x,ksize=[1,2,2,2,1],strides=[1,2,2,2,1],padding="SAME")

    def _get_conv_variable(self,input_size):
        out = tf.Variable(tf.truncated_normal(input_size,stddev=0.01,name="weights"))
        return out

    def _get_bias_variable(self,input_size):
        out = tf.Variable(tf.zeros(input_size),name="biases")
        return out

    def train_model(self,model,train_data,train_label,real_epochs):

        model_checkpoint = ModelCheckpoint('unet.hdf5',
                                           monitor='loss',
                                           save_best_only=True)
        train_hist = model.fit(train_data,train_label,
                               batch_size=self.batch_size,
                               epochs=self.epochs,
                               validation_split=0.1,
                               shuffle=True,
                               callbacks=[model_checkpoint],
                               verbose=1)
        if real_epochs == 1:
            print "[*] saving model weights"
            model.save_weights('model_weights.h5',overwrite=True)

        return model,train_hist

    def test_model(self,model,test_data):
        model.load_weights('model_weights.h5')
        print "[*] begin to test learned model"
        denoised = model.predict(test_data,verbose=1)
        return denoised