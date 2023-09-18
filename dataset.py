from glob2 import glob
from natsort import natsorted
import numpy as np
import keras
import cv2

def get_broken_labels(broken_dir):
    broken_list = glob(broken_dir+"/**/*.jpg")
    broken_list = natsorted(broken_list)
    # broken_broken_labels, broken_dirty_onehots = get_dirty_label_from_list(broken_list)
    return broken_list, get_broken_labels_from_list(broken_list)

def get_dirty_labels(dirty_dir):
    dirty_list = glob(dirty_dir+"/*.jpg")
    dirty_list = natsorted(dirty_list)
    return dirty_list, get_dirty_labels_from_list(dirty_list)

def get_broken_labels_from_list(broken_list):
    broken_broken_labels = np.ones(len(broken_list))
    broken_dirty_onehots = np.zeros((len(broken_list), 7))
    broken_dirty_onehots[:, 0] += 1 # [1, 0, 0, 0, 0, 0, 0]
    return broken_broken_labels, broken_dirty_onehots

def get_dirty_labels_from_list(dirty_list):
    dirty_broken_labels = np.zeros(len(dirty_list))
    dirty_lbl_list = [name.replace(".jpg", ".txt") for name in dirty_list]
    dirty_lbl_list = np.array(dirty_lbl_list)
    _, labels_eigen_vector = np.linalg.eig(np.diag((1,2,3,4,5,6)))
    dirty_dirty_onehots = np.array([labels_eigen_vector[int(open(lbl_name, 'r').readlines()[0])] for lbl_name in dirty_lbl_list])
    dirty_dirty_onehots = np.hstack((np.zeros(dirty_dirty_onehots.shape[0]).reshape(-1, 1), dirty_dirty_onehots))
    return dirty_broken_labels, dirty_dirty_onehots

def get_merge_datas(broken_labels, dirty_labels):
    total_image_list = np.append(broken_labels[0], dirty_labels[0])
    total_broken_labels = np.append(broken_labels[1][0], dirty_labels[1][0])
    total_dirty_labels = np.concatenate([broken_labels[1][1], dirty_labels[1][1]])
    return total_image_list, total_broken_labels, total_dirty_labels
    

class DataGenerator(keras.utils.Sequence):
    def __init__(self, data, dirty_labels, batch_size, model_input_shape, shuffle=True):
        self.model_input_shape = model_input_shape
        self.input_shape = self.model_input_shape[1:]
        self.input_size = self.input_shape[:-1]
    
        self.data = data
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
        batch_dirty_labels = self.dirty_labels[batch_indexes]
        
        return batch_imgs, batch_dirty_labels
    
    def get_batch_mask(self):
        return self.batch_mask
    
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)