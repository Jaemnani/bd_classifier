import tensorflow as tf
import keras
from tensorflow.keras.applications import efficientnet_v2, EfficientNetV2B0 

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
    
    return tf.keras.Model(feature_extractor.input, total_output, name=base_model.name+"_db_"+function)

if __name__ == '__main__':
    model = get_bd_model()
    print(model.summary())