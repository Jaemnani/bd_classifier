import os
import numpy as np
import keras
import cv2
from glob2 import glob
from natsort import natsorted
import tensorflow as tf
from train import set_gpu_memory_growth

set_gpu_memory_growth()

img_list = glob("./test_images/*.jpg")
img_list = natsorted(img_list)

model = keras.models.load_model("./outputs/efficientnetv2-b0_db_softmax/efficientnetv2-b0_db_softmax.h5")

input_shape = model.input_shape[1:]
input_size = input_shape[:-1]

dummy = np.zeros(input_shape)
dummy = np.expand_dims(dummy, axis=0)
model.predict(dummy)

for img_idx, img_path in enumerate(img_list):
    print(os.path.basename(img_path), " : " , end="")
    img = cv2.imread(img_path)
    img = cv2.resize(img, input_size)
    img = np.expand_dims(img, axis=0)
    out = model.predict(img, verbose=0)
    
    isBroken = False
    isDirty  = False
    result_state = np.argmax(out)
    result_score = np.max(out)
    if result_state == 0:
        isBroken = True
    else:
        if result_state > 3:
            isDirty = True

    if isBroken == True:
        print("broken score(%.3f)"%(result_score))
    else:
        print("dirty(%d) score(%.3f)"%(result_state, result_score))
    



