import tensorflow as tf
import keras
from tensorflow.keras.applications import efficientnet_v2, EfficientNetV2B0 

# global save_path
# g_save_path = ""
# def set_save_path(save_path):
#     global g_save_path
#     g_save_path = save_path
#     print("set Global save path is", g_save_path)
    
# es_callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)
# mc_callback = tf.keras.callbacks.ModelCheckpoint(file_path=g_save_path + "/checkpoints", save_weights_only=True, monitor='val_accuracy', mode='max', save_best_only=True)

def get_callbacks(save_path):
    callbacks = []
    callbacks.append(tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3))
    callbacks.append(tf.keras.callbacks.ModelCheckpoint(filepath=save_path + "/checkpoints/", save_weights_only=True, monitor='val_accuracy', mode='max', save_best_only=True))
    return callbacks

def get_bd_model(num_classes=7, function="softmax", freeze=True):
    base_model = EfficientNetV2B0(weights='imagenet')
    feature_extractor = tf.keras.Model(inputs=base_model.input, outputs=base_model.get_layer('top_activation').output, name=base_model.name+"_FE")
    if freeze == True:
        feature_extractor.trainable = False
    dx1 = tf.keras.layers.Conv2D(64, (3,3), activation="relu")(feature_extractor.output)
    dx2 = tf.keras.layers.MaxPooling2D()(dx1)
    dx3 = tf.keras.layers.Flatten()(dx2)
    dx4 = tf.keras.layers.Dense(128, activation='relu')(dx3)
    dd1 = tf.keras.layers.Dropout(rate=0.2)(dx4)
    
    if not function == "softmax":
        function = "sigmoid"
        print("Function error, set sigmoid function.")
    total_output = tf.keras.layers.Dense(7, activation=function, name="output")(dd1)
    
    return tf.keras.Model(feature_extractor.input, total_output, name=base_model.name+"db_"+function)

def set_model_compile(model):
    model.compile(
        optimizer = keras.optimizers.Adam(),
        loss = keras.losses.CategoricalCrossentropy(),
        metrics='accuracy',
    )



class BD_Model(tf.keras.Model):
    def __init__(self, num_classes=7, freeze=True):
        super(BD_Model, self).__init__()
        self.base_model = EfficientNetV2B0(weights='imagenet')
        
        if freeze:
            self.base_model.trainable = False
        

class DirtyModel(tf.keras.Model):
    def __init__(self, num_classes=6, freeze=True):
        super(DirtyModel, self).__init__()
        self.base_model = EfficientNetV2B0(weights='imagenet')

        if freeze:
            self.base_model.trainable = False
        
        self.top = tf.keras.Sequential([
            tf.keras.layers.GlobalAveragePooling2D(name='avg_pool'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.2, name="top_dropout"),
        ])

        self.classifier = tf.keras.layers.Dense(num_classes, activation='softmax' , name='output_dirty')

        self.build(input_shape=(None, 224, 224, 3))

    def call(self, inputs, training=True):

        x = self.base_model(inputs)
        x = self.top(x)
        x = self.classifier(x)
        return x

if __name__ == '__main__':
    model = get_bd_model()
    # model.build(input_shape=(None, 224, 224, 3))
    print(model.summary())