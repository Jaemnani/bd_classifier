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

lbl_list = [name.replace(".jpg", ".txt") for name in img_list]
dirty_lbl_list = np.array(lbl_list)
_, labels_eigen_vector = np.linalg.eig(np.diag((1,2,3,4,5,6,7)))

actual = []
for idx, lbl_name in enumerate(dirty_lbl_list):
    if os.path.exists(lbl_name):
        dirty_label = int(open(lbl_name, 'r').readlines()[0])
    else:
        dirty_label = -1
    # import pdb; pdb.set_trace()
    onehot = labels_eigen_vector[dirty_label+1]
    actual.append(onehot)
actual = np.array(actual)

model = keras.models.load_model("./outputs/efficientnetv2-b0_db_softmax_e30+f10/efficientnetv2-b0_db_softmax.h5")
# model = keras.models.load_model("./outputs/efficientnetv2-b0_db_softmax_02/efficientnetv2-b0_db_softmax.h5")

input_shape = model.input_shape[1:]
input_size = input_shape[:-1]

dummy = np.zeros(input_shape)
dummy = np.expand_dims(dummy, axis=0)
model.predict(dummy)

imgs = np.array([cv2.resize(cv2.imread(name), input_size) for name in img_list])
preds = model.predict(imgs, verbose=0)

mPrecision = keras.metrics.Precision()
mRecall = keras.metrics.Recall()
mF1None = keras.metrics.F1Score(average=None)
mF1Micro = keras.metrics.F1Score(average='micro')
mF1Macro = keras.metrics.F1Score(average='macro')
mF1Weighted = keras.metrics.F1Score(average="weighted")

mPrecision.update_state(actual, preds)
mRecall.update_state(actual, preds)
mF1None.update_state(actual, preds)
mF1Micro.update_state(actual, preds)
mF1Macro.update_state(actual, preds)
mF1Weighted.update_state(actual, preds)

print("Precision : ", mPrecision.result())
print("Recall    : ", mRecall.result())
print("F1 None   : ", mF1None.result())
print("F1 Micro  : ", mF1Micro.result())
print("F1 Macro  : ", mF1Macro.result())
print("f1 weight : ", mF1Weighted.result())

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
        print("dirty(%d) score(%.3f)"%(result_state-1, result_score))
    



print("done")