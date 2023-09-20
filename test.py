import os
import numpy as np
import keras
import cv2
import argparse
from glob2 import glob
from natsort import natsorted
# import tensorflow as tf
from train import set_gpu_memory_growth
from dataset import get_broken_labels, get_dirty_labels, get_merge_datas
from sklearn.model_selection import train_test_split


def test(opt):
    set_gpu_memory_growth()
    broken_labels = get_broken_labels(opt.broken_dir)
    dirty_labels = get_dirty_labels(opt.dirty_dir)
    total_image_list, total_dirty_labels = get_merge_datas(broken_labels, dirty_labels)

    train_X, test_x, train_Y, test_y = train_test_split(total_image_list, total_dirty_labels, test_size=2000./len(total_image_list), random_state=42)
    # train_x, valid_x, train_y, valid_y = train_test_split(train_X, train_Y, test_size = 0.1, random_state=42)
    
    # img_list = glob("./test_images/*.jpg") # # for test 22 images
    img_list = test_x
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

    # model = keras.models.load_model("./outputs/efficientnetv2-b0_db_softmax_e30+f10_full/efficientnetv2-b0_db_softmax.h5")
    model = keras.models.load_model("./outputs/efficientnetv2-b0_db_softmax_e30+f10_5876ea/efficientnetv2-b0_db_softmax.h5")
    # model = keras.models.load_model("./outputs/efficientnetv2-b0_db_softmax_04_quant_default/efficientnetv2-b0_db_softmax.h5")
    

    input_shape = model.input_shape[1:]
    input_size = input_shape[:-1]

    dummy = np.zeros(input_shape)
    dummy = np.expand_dims(dummy, axis=0)
    model.predict(dummy)

    imgs = np.array([cv2.resize(cv2.imread(name), input_size) for name in img_list])
    preds = model.predict(imgs, verbose=0)

    mPrecision = keras.metrics.Precision()
    mRecall = keras.metrics.Recall()
    # mF1None = keras.metrics.F1Score(average=None)
    mF1Micro = keras.metrics.F1Score(average='micro')
    mF1Macro = keras.metrics.F1Score(average='macro')
    mF1Weighted = keras.metrics.F1Score(average="weighted")

    mPrecision.update_state(actual, preds)
    mRecall.update_state(actual, preds)
    # mF1None.update_state(actual, preds)
    mF1Micro.update_state(actual, preds)
    mF1Macro.update_state(actual, preds)
    mF1Weighted.update_state(actual, preds)

    print("\n * Softmax Classifier Accuracy")
    print("Precision : ", mPrecision.result())
    print("Recall    : ", mRecall.result())
    # print("F1 None   : ", mF1None.result())
    print("F1 Micro  : ", mF1Micro.result())
    print("F1 Macro  : ", mF1Macro.result())
    print("f1 weight : ", mF1Weighted.result())

    mPrecision = keras.metrics.Precision()
    mRecall = keras.metrics.Recall()
    mF1None = keras.metrics.F1Score(average=None)

    actual_change = np.array([True if pred.argmax()==0 or pred.argmax() > 2 else False for pred in actual]).reshape(-1,1).astype(int)
    pred_change = np.array([True if pred.argmax()==0 or pred.argmax() > 2 else False for pred in preds]).reshape(-1,1).astype(int)
    
    mPrecision.update_state(actual_change, pred_change)
    mRecall.update_state(actual_change, pred_change)
    mF1None.update_state(actual_change, pred_change)

    print("\n * Need Changed Accuracy")
    print("Precision : ", mPrecision.result())
    print("Recall    : ", mRecall.result())
    print("F1 Score   : ", mF1None.result())

def parser_opt(): 
    parser = argparse.ArgumentParser()
    parser.add_argument("--broken_dir", dest="broken_dir", type=str, default="./dataset/datasets_broken/")
    parser.add_argument("--dirty_dir", dest="dirty_dir", type=str, default="./dataset/datasets_dirty2/")
    # parser.add_argument("--batch_size", dest="batch_size", type=int, default=1, help="total batch size")
    # parser.add_argument("--save_dir", dest="save_dir", type=str, default="./outputs/")
    return parser.parse_args()

if __name__=="__main__":
    opt = parser_opt()
    test(opt)