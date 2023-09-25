import tensorflow as tf
from tensorflow.keras.applications import EfficientNetV2B0 
import numpy as np

class DirtyModel(tf.keras.Model):
    def __init__(self, num_classes=7, function="softmax", freeze=True):
        super(DirtyModel, self).__init__()
        self.base_model = ''
        self.feature_extractor = ''
        self.num_classes = num_classes
        self.function = function
        self.freeze = freeze
        self.model = self.build_model(self.num_classes, self.function, self.freeze)
        self._init_set_name(self.model.name)

    def get_dummy_shape(self):
        shape = np.array(self.model.input_shape)
        shape[0] = 1
        return tuple(shape)
        
    def build_model(self, num_classes, function, freeze):
        self.base_model = EfficientNetV2B0(weights='imagenet')
        self.feature_extractor = tf.keras.Model(inputs=self.base_model.input, outputs=self.base_model.get_layer('top_activation').output, name=self.base_model.name + "_FE")

        if freeze:
            self.feature_extractor.trainable = False

        dx1 = tf.keras.layers.Conv2D(64, (3, 3), activation="relu", padding='same')(self.feature_extractor.output)
        dx2 = tf.keras.layers.MaxPooling2D()(dx1)
        dx3 = tf.keras.layers.Flatten()(dx2)
        dx4 = tf.keras.layers.Dense(128, activation='relu')(dx3)
        dd1 = tf.keras.layers.Dropout(rate=0.2)(dx4)
        total_output = tf.keras.layers.Dense(num_classes, activation=function, name="output")(dd1)

        model = tf.keras.Model(self.feature_extractor.input, total_output, name=self.base_model.name + "_db_" + function)
        return model

    def call(self, inputs):
        return self.model(inputs)
    
    def save_model(self, save_path):
        self.model.save(save_path +"/"+ self.model.name + ".h5")
        
        
if __name__ == '__main__':
    pass