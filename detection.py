from network import ContourGan
from keras.preprocessing import image
from PIL import Image
from util import *
import os
from scipy import misc

model = ContourGan(256, 256).build_generator()
model.load_weights('model/weights_canny.h5')
imgs = os.listdir('test')
for im in imgs:
    img = image.load_img(os.path.join('test', im))
    img = img.resize((256, 256), Image.NEAREST).convert('RGB')
    img = image.img_to_array(img)
    img = normalize(img)
    res = model.predict(np.expand_dims(img, axis=0))
    res_np = res.astype(np.float32)
    cond = np.greater_equal(res_np, 0.95).astype(np.int)
    misc.imsave(os.path.join("output", im.split('.')[0] + '.png'), cond[0, :, :, 0])
