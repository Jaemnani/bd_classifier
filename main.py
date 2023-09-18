import tensorflow as tf
import keras
import numpy as np
from glob import glob
import cv2
import os
from sklearn.model_selection import train_test_split
from natsort import natsorted
import keras.backend as K
from keras.callbacks import Callback
import matplotlib.pyplot as plt

from dataset import *
from train import *
from model import *

set_gpu_memory_limit()

model = tf.keras.applications.efficientnet_v2.EfficientNetV2B0()
model = DirtyModel()
dirty_img_list, dirty_labels_onehot = data_load_dirty("./dataset/datasets_dirty2/")

loss_function = tf.keras.losses.CategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam()
train_acc_metric = tf.keras.metrics.CategoricalAccuracy()

d_train_x, d_valid_x, d_train_y, d_valid_y = train_test_split(dirty_img_list, dirty_labels_onehot, test_size=0.2, random_state=42)

train_datas = DataGenerator_dirty(d_train_x, d_train_y, 32, model.input_shape)
valid_datas = DataGenerator_dirty(d_valid_x, d_valid_y, 32, model.input_shape)

# feature_extractor= tf.keras.Model(inputs=model.input, outputs=model.get_layer('avg_pool').output, name=model.name+"_FE")
feature_extractor= tf.keras.Model(inputs=model.input, outputs=model.get_layer('top_activation').output, name=model.name+"_FE")
feature_extractor.trainable = False

# Adding Self-Attention layer
x1 = tf.keras.layers.Conv2D(64, (3,3), activation="relu")(feature_extractor.output)
x2 = tf.keras.layers.MaxPooling2D()(x1)
x3 = tf.keras.layers.Flatten()(x2)
x4 = tf.keras.layers.Dense(128, activation='relu')(x3)
# attention_layer = SelfAttention(filters=32)(x4)
d1 = tf.keras.layers.Dropout(rate=0.2)(x4)
# dirty_output = tf.keras.layers.Dense(6, activation='softmax', name='output_dirty')(attention_layer)

dirty_output = tf.keras.layers.Dense(6, activation='softmax', name='output_dirty')(d1)
new_model = tf.keras.Model(feature_extractor.input, dirty_output, name=model.name+"_test_dirty")

K.set_floatx("float16")
K.set_epsilon(1e-4)

try:
    # # Training
    save_path = "./outputs/"+new_model.name + "/"
    print("save_path is ", save_path)
    if not os.path.exists(save_path):
        os.makedirs(save_path + "/checkpoints/")


    new_model.compile(
        optimizer=keras.optimizers.Adam(clipnorm=1.0, clipvalue=0.5), 
        loss={'output_dirty':keras.losses.CategoricalCrossentropy()},
        metrics={'output_dirty':keras.metrics.CategoricalAccuracy()},
        )

    # new_model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.SGD(momentum=0.9, clipnorm=1.0, clipvalue=0.5), metrics='accuracy')
    # new_model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.RMSprop(clipnorm=1.0, clipvalue=0.5), metrics='accuracy')

    # # # Model Check point    
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=save_path+"checkpoints/", save_weights_only=True, monitor='val_accuracy', mode='max', save_best_only=True)
    # # # Learning rate Scheduler
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=5, min_lr=0.0001)
    # lr_finder = LearningRateFinder(start_lr=1e-8, end_lr=1e-2, num_steps=500)
    # # # EarlyStopping
    es_callback = tf.keras.callbacks.EarlyStopping(monitor="loss", patience=3 )

    new_model.fit(train_datas, epochs=EPOCH, validation_data=valid_datas, callbacks=[model_checkpoint_callback, es_callback])

    # # Fine Tuning
    new_model.trainable = True
    model.compile(optimizer=keras.optimizers.Adam(1e-5, clipnorm=1.0, clipvalue=0.5),
                # optimizer=keras.optimizers.SGD(1e-5),
                # optimizer=keras.optimizers.RMSprop(1e-5), 
                loss={'output_dirty':keras.losses.CategoricalCrossentropy()},
                metrics={'output_dirty':keras.metrics.CategoricalAccuracy()},
                )

    new_model.fit(train_datas, epochs=FINETUNE_EPOCH, validation_data=valid_datas, callbacks=[model_checkpoint_callback, es_callback])
except Exception as e:
    print(e)
    new_model.save(save_path + new_model.name + ".h5")
new_model.save(save_path + new_model.name +".h5")

print("done")


# import tensorflow as tf
# import keras
# import numpy as np
# from glob import glob
# import cv2
# import os
# from sklearn.model_selection import train_test_split
# from natsort import natsorted
# import keras.backend as K

# from train import *
# from dataset import *

# set_gpu_memory_limit()

# model = tf.keras.applications.efficientnet_v2.EfficientNetV2B0()

# # # Training data load
# broken_list = get_img_list(broken_dir)


# # broken_list = glob("./dataset/datasets_broken/**/*.jpg")[::10]
# # broken_list = natsorted(broken_list)
# broken_broken_labels = np.ones(len(broken_list))
# broken_dirty_onehot = np.zeros((len(broken_list), 6))
# broken_dirty_onehot[:, -1] += 1

# # normal_list == dirty_img_list
# normal_list = glob("./dataset/datasets_dirty2/*.jpg")
# normal_list = natsorted(normal_list)
# normal_broken_labels = np.zeros(len(normal_list))

# dirty_lbl_list = [name.replace(".jpg", ".txt") for name in normal_list]
# dirty_lbl_list = np.array(dirty_lbl_list)
# labels_eigenvalue, labels_eigenvector = np.linalg.eig(np.diag((1,2,3,4,5,6)))
# normal_dirty_onehot = np.array([labels_eigenvector[int(open(lbl_name, 'r').readlines()[0])] for lbl_name in dirty_lbl_list])

# total_image_list = np.append(broken_list, normal_list)
# total_broken_labels = np.append(broken_broken_labels, normal_broken_labels)
# total_dirty_labels = np.concatenate([broken_dirty_onehot, normal_dirty_onehot])

# train_x, valid_x, train_y1, valid_y1, train_y2, valid_y2 = train_test_split(total_image_list, total_broken_labels, total_dirty_labels, test_size = 0.2, random_state=42 )
# # # b_train_x, b_valid_x, b_train_y, b_valid_y, b_d_train_y, b_d_valid_y = train_test_split(broken_list, broken_labels, broken_dirty_labels_onehot, test_size = 0.2, random_state=42)
# # # d_train_x, d_valid_x, d_train_y, d_valid_y, d_b_train_y, d_b_valid_y = train_test_split(dirty_list, dirty_labels_onehot, dirty_broken_labels, test_size=0.2, random_state=42)

# # total_train_x = np.append(b_train_x, d_train_x)
# # total_valid_x = np.append(b_valid_x, d_valid_x)
# # total_train_b_y = np.append(b_train_y, d_b_train_y)
# # total_valid_b_y = np.append(b_valid_y, d_b_valid_y)
# # total_train_d_y = np.concatenate([b_d_train_y, d_train_y])
# # total_valid_d_y = np.concatenate([b_d_valid_y, d_valid_y])

# train_datas = DataGenerator(train_x, train_y1, train_y2, 32, model.input_shape)
# valid_datas = DataGenerator(valid_x, valid_y1, valid_y2, 32, model.input_shape)

# # feature_extractor= tf.keras.Model(inputs=model.input, outputs=model.get_layer('avg_pool').output, name=model.name+"_FE")
# # feature_extractor.trainable = False
# feature_extractor= tf.keras.Model(inputs=model.input, outputs=model.get_layer('top_activation').output, name=model.name+"_FE")
# feature_extractor.trainable = False

# dirty_x1 = tf.keras.layers.Conv2D(64, (3,3), activation="relu")(feature_extractor.output)
# dirty_x2 = tf.keras.layers.MaxPooling2D()(dirty_x1)
# dirty_x3 = tf.keras.layers.Flatten()(dirty_x2)
# dirty_x4 = tf.keras.layers.Dense(128, activation='relu')(dirty_x3)
# dirty_d1 = tf.keras.layers.Dropout(rate=0.2)(dirty_x4)
# dirty_output = tf.keras.layers.Dense(6, activation='softmax', name='output_dirty')(dirty_d1)

# broken_x1 = tf.keras.layers.GlobalAveragePooling2D()(feature_extractor.output)
# broken_output = tf.keras.layers.Dense(1, activation='sigmoid', name='output_broken')(broken_x1)
# # broken_output = tf.keras.layers.Dense(1, activation='sigmoid', name='output_broken')(dirty_d1)


# # # Training
# new_model = tf.keras.Model(feature_extractor.input, [broken_output, dirty_output], name=model.name+"_broken_dirty")


# save_path = "./outputs/" + new_model.name + "/"
# if not os.path.exists(save_path):
#     os.makedirs(save_path + "/checkpoints/")

# model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=save_path+"checkpoints/",)

# def custom_loss(y_true, y_pred):
#     broken_loss = tf.losses.categorical_crossentropy(y_true[0], y_pred[0])
#     dirty_loss = tf.losses.binary_crossentropy(y_true[1], y_pred[1])

#     dirty_loss = dirty_loss * (1-y_true[0])

#     # if y_true[0] > 0.5:
#     #     dirty_loss = dirty_loss * (1-y_true[0])
    
#     return broken_loss + dirty_loss



# new_model.compile(
#     optimizer = keras.optimizers.Adam(), 
#     loss=custom_loss,
#     metrics=['accuracy'],
#     )

# new_model.fit(train_datas, epochs=EPOCH, validation_data=valid_datas, callbacks=[model_checkpoint_callback])



# # # # Fine Tuning
# # new_model.trainable = True
# # model.compile(optimizer=keras.optimizers.Adam(1e-5),
# #             loss={'output_broken':keras.losses.BinaryCrossentropy(), 'output_dirty':keras.losses.CategoricalCrossentropy()},
# #             metrics={'output_broken':keras.metrics.BinaryAccuracy(), 'output_dirty':keras.metrics.CategoricalAccuracy()}, 
# #               )
# # # new_model.fit(total_train_x, {'output_broken':total_train_b_y, "output_dirty":total_train_d_y}, epochs=FINETUNE_COUNT, validation_data=(total_valid_x, {'output_broken':total_valid_b_y, 'output_dirty':total_valid_d_y}) , callbacks=[model_checkpoint_callback])
# # new_model.fit(train_datas, epochs=EPOCH, validation_data=valid_datas, callbacks=[model_checkpoint_callback])

# # new_model.save(save_path + new_model.name)
# new_model.save(save_path + new_model.name +".h5")

# print("done")
