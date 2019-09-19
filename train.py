from network import *
from util import *
from keras.preprocessing.image import ImageDataGenerator
import os
import numpy as np
from scipy import misc


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#config = tf.ConfigProto()
#config.gpu_options.per_process_gpu_memory_fraction = 0.3
#set_session(tf.Session(config=config))


total_imgs = 200
batch_size = 1
n_epochs = 50
n_steps = total_imgs // batch_size * n_epochs


def train(d_model, g_model, gan_model, data_generator, w_patch=16, h_patch=16):
    pre_gen_loss = 0.0
    i = 0
    valid = np.ones(shape=(batch_size, h_patch, w_patch, 1))
    fake = np.zeros(shape=(batch_size, h_patch, w_patch, 1))
    for batch in data_generator:
        if i == n_steps:
            break
        img = batch[0]
        label_real = batch[1]
        label_fake = g_model.predict(img)
        d_loss_real = d_model.train_on_batch(label_real, valid)
        d_loss_fake = d_model.train_on_batch(label_fake, fake)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
        g_loss = gan_model.train_on_batch(img, [label_real, valid])
        # summarize performance
        print('>%d, d1[%.9f] d2[%.9f] g[%.9f] acc1[%.9f] acc2[%.9f]' % (i + 1, d_loss_real[0], d_loss_fake[0], g_loss[0], d_loss_real[1], d_loss_fake[1]))
        if i == 0:
            pre_gen_loss = g_loss
        else:
            if g_loss < pre_gen_loss:
                pre_gen_loss = g_loss
                g_model.save_weights("model/weights_g.h5", overwrite=True)
        if i % 200 == 0:
            res_np = label_fake.astype(np.float32)
            cond = np.greater_equal(res_np, 0.5).astype(np.int)
            misc.imsave(os.path.join("output", str(i) + '.png'), cond[0, :, :, 0])
        i = i + 1


train_img_generator = ImageDataGenerator(preprocessing_function=normalize)
train_img = train_img_generator.flow_from_directory(directory="BSDS/train/imgs",
                                                    target_size=(256, 256),
                                                    color_mode='rgb',
                                                    class_mode=None,
                                                    batch_size=batch_size,
                                                    shuffle=True,
                                                    seed=1
                                                    )

train_label_generator = ImageDataGenerator(preprocessing_function=normalize)
train_label = train_label_generator.flow_from_directory(directory="BSDS/train/mask",
                                                        target_size=(256, 256),
                                                        color_mode='grayscale',
                                                        class_mode=None,
                                                        batch_size=batch_size,
                                                        shuffle=True,
                                                        seed=1
                                                        )

data_generator = zip(train_img, train_label)
contour_gan = ContourGan(256, 256)
d = contour_gan.build_discriminator()
g = contour_gan.build_generator()
gan = contour_gan.contour_gan(g, d)
d.load_weights("model/weights_d.h5")
g.load_weights("model/weights_g.h5")
train(d, g, gan, data_generator)
d.save('model_d.h5')
g.save('model/model_g.h5')


