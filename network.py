from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
from keras.layers import Input, Concatenate
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Model
from keras.optimizers import Adam
from keras.applications.vgg16 import VGG16
from util import *


class ContourGan:

    def __init__(self, height, weight, channels=3):
        self.height = height
        self.weight = weight
        self.channels = channels
        self.img_shape = (self.height, self.weight, self.channels)

    def build_generator(self):
        # U-net like
        input_img = Input(name='input_img',
                          shape=(self.height,
                                 self.weight,
                                 self.channels),
                          dtype='float32')
        vgg16 = VGG16(input_tensor=input_img,
                      weights='imagenet',
                      include_top=False)
        vgg_pools = [vgg16.get_layer('block%d_pool' % i).output
                     for i in range(1, 6)]

        def decoder(layer_input, skip_input, channel, last_block=False):
            if not last_block:
                concat = Concatenate(axis=-1)([layer_input, skip_input])
                bn1 = InstanceNormalization()(concat)
            else:
                bn1 = InstanceNormalization()(layer_input)
            conv_1 = Conv2D(channel, 1,
                            activation='relu', padding='same')(bn1)
            bn2 = InstanceNormalization()(conv_1)
            conv_2 = Conv2D(channel, 3,
                            activation='relu', padding='same')(bn2)
            return conv_2

        d1 = decoder(UpSampling2D((2, 2))(vgg_pools[4]), vgg_pools[3], 256)
        d2 = decoder(UpSampling2D((2, 2))(d1), vgg_pools[2], 128)
        d3 = decoder(UpSampling2D((2, 2))(d2), vgg_pools[1], 64)
        d4 = decoder(UpSampling2D((2, 2))(d3), vgg_pools[0], 32)
        d5 = decoder(UpSampling2D((2, 2))(d4), None, 32, True)

        output = Conv2D(1, 3, activation='sigmoid', padding='same')(d5)
        model = Model(inputs=input_img, outputs=[output])
        # model.summary()
        return model

    def build_discriminator(self):

        def d_layer(layer_input, filters, f_size=4, normalization=True):
            d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
            d = LeakyReLU(alpha=0.2)(d)
            if normalization:
                d = InstanceNormalization()(d)
            return d

        image = Input(shape=(self.height, self.weight, 1))

        d1 = d_layer(image, 64, normalization=False)
        d2 = d_layer(d1, 128)
        d3 = d_layer(d2, 256)
        d4 = d_layer(d3, 512)

        patch_out = Conv2D(1, kernel_size=4, strides=1, padding='same')(d4)
        discriminator = Model(image, patch_out)
        optimizer = Adam(0.0002, 0.5)
        discriminator.compile(loss='mse', optimizer=optimizer, metrics=['accuracy'])
        # discriminator.summary()
        return discriminator

    def contour_gan(self, g, d):
        d.trainable = False
        in_src = Input(shape=self.img_shape)
        g_out = g(in_src)
        d_out = d(g_out)
        gan = Model(in_src, [g_out, d_out])
        opt = Adam(lr=1E-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
        gan.compile(loss=[balance_loss, 'mse'], optimizer=opt, loss_weights=[10, 1])
        return gan
