from glob2 import glob
from natsort import natsorted
import numpy as np
import keras
import cv2

broken_dir = "./train/broken_totals/**/"
normal_dir = "./train/train_datasets_dirty2/"

def get_img_list(dir):
    if "/*.jpg" in dir:
        file_list = glob(dir)
    else:
        file_list = glob(dir + "/*.jpg")
    file_list = natsorted(file_list)
    return file_list

def get_label_dirty(img_list):
    lbl_list = np.array([name.replace(".jpg", ".txt") for name in img_list])
    _, labels_eigen_vector = np.linalg.eig(np.diag((1,2,3,4,5,6)))
    lbl_onehot = np.array([int(open(lbl_name, 'r').readlines()[0]) for lbl_name in lbl_list])
    return lbl_onehot

def get_label_dirty_onehot(img_list):
    lbl_list = np.array([name.replace(".jpg", ".txt") for name in img_list])
    _, labels_eigen_vector = np.linalg.eig(np.diag((1,2,3,4,5,6)))
    lbl_onehot = np.array([labels_eigen_vector[int(open(lbl_name, 'r').readlines()[0])] for lbl_name in lbl_list])
    return lbl_onehot

def data_load_dirty(data_path):
    dirty_img_list = get_img_list(data_path)
    dirty_labels_onehot = get_label_dirty(dirty_img_list)
    return dirty_img_list, dirty_labels_onehot

class DataGenerator(keras.utils.Sequence):
    def __init__(self, data, broken_labels, dirty_labels, batch_size, model_input_shape, shuffle=True):
        self.model_input_shape = model_input_shape
        self.input_shape = self.model_input_shape[1:]
        self.input_size = self.input_shape[:-1]
    
        self.data = data
        # self.labels = labels
        self.broken_labels = broken_labels
        self.dirty_labels = dirty_labels
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indexes = np.arange(len(self.data))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __len__(self):
        return int(np.ceil(len(self.data) / self.batch_size))
    
    def __getitem__(self, idx):
        start_idx = idx * self.batch_size
        end_idx = (idx + 1) * self.batch_size
        batch_indexes = self.indexes[start_idx : end_idx]

        batch_data = self.data[batch_indexes]
        batch_imgs = [ cv2.resize( cv2.imread(name), self.input_size) for name in self.data[batch_indexes]]
        batch_imgs = np.array(batch_imgs)

        batch_broken_labels = self.broken_labels[batch_indexes].reshape(-1,1)
        batch_dirty_labels = self.dirty_labels[batch_indexes]

        return batch_imgs, [batch_broken_labels, batch_dirty_labels]
        
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)


class DataGenerator_dirty(keras.utils.Sequence):
    def __init__(self, data, dirty_labels, batch_size, model_input_shape, shuffle=True):
        self.model_input_shape = model_input_shape
        self.input_shape = self.model_input_shape[1:]
        self.input_size = self.input_shape[:-1]
    
        self.data = data
        # self.labels = labels
        # self.broken_labels = broken_labels
        self.dirty_labels = dirty_labels
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indexes = np.arange(len(self.data))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __len__(self):
        return int(np.ceil(len(self.data) / self.batch_size))
    
    def __getitem__(self, idx):
        start_idx = idx * self.batch_size
        end_idx = (idx + 1) * self.batch_size
        batch_indexes = self.indexes[start_idx : end_idx]

        batch_data = self.data[batch_indexes]
        batch_imgs = [ cv2.resize( cv2.imread(name), self.input_size) for name in self.data[batch_indexes]]
        batch_imgs = np.array(batch_imgs)

        # batch_broken_labels = self.broken_labels[batch_indexes].reshape(-1,1)
        batch_dirty_labels = self.dirty_labels[batch_indexes]

        # return batch_imgs, [batch_broken_labels, batch_dirty_labels]
        return batch_imgs, batch_dirty_labels
        
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)



    