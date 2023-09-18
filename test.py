import tensorflow as tf
from tensorflow.keras.applications import EfficientNetV2B0

class YogaPose(tf.keras.Model):
    def __init__(self, num_classes=30, freeze=False):
        super(YogaPose, self).__init__()
        self.base_model = EfficientNetV2B0(include_top=False, weights='imagenet')

        # Freeze the pretrained weights
        if freeze:
            self.base_model.trainable = False

        self.top = tf.keras.Sequential([tf.keras.layers.GlobalAveragePooling2D(name="avg_pool"),
                                       tf.keras.layers.BatchNormalization(),
                                       tf.keras.layers.Dropout(0.5, name="top_dropout")])
        self.classifier = tf.keras.layers.Dense(num_classes, activation="softmax", name="pred")

    def call(self, inputs, training=True):
        x = self.base_model(inputs)
        x = self.top(x)
        x = self.classifier(x)
        return x


if __name__ == '__main__':
    model = YogaPose(num_classes=107, freeze=True)
    model.build(input_shape=(None, 224, 224, 3))
    print(model.summary())