import tensorflow as tf
from tensorflow.keras.applications import efficientnet_v2
from tensorflow.keras.applications import EfficientNetV2B0

class SelfAttention(tf.keras.layers.Layer):
    def __init__(self, filters):
        super(SelfAttention, self).__init__()
        self.filters = filters

    def build(self, input_shape):
        # self.W_q = self.add_weight(shape=(input_shape[-1], input_shape[-1]), initializer='glorot_uniform', trainable=True, name='W_q')
        # self.W_k = self.add_weight(shape=(input_shape[-1], input_shape[-1]), initializer='glorot_uniform', trainable=True, name='W_k')

        self.W_q = self.add_weight(name='W_q', shape=(input_shape[-1], self.filters), initializer='uniform', trainable=True)
        self.W_k = self.add_weight(name='W_k', shape=(input_shape[-1], self.filters), initializer='uniform', trainable=True)
        self.W_v = self.add_weight(name='W_v', shape=(input_shape[-1], input_shape[-1]), initializer='uniform', trainable=True)

    def call(self, inputs):
        q = tf.matmul(inputs, self.W_q)
        k = tf.matmul(inputs, self.W_k)
        v = tf.matmul(inputs, self.W_v)
        
        attention_scores = tf.matmul(q, k, transpose_b=True)
        # attention_scores = attention_scores / tf.math.sqrt(tf.cast(inputs.shape[-1], dtype=tf.float16))
        attention_scores = tf.nn.softmax(attention_scores, axis=-1)
        
        # output = tf.matmul(attention_scores, inputs)
        output = tf.matmul(attention_scores, v)
        
        return output

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
    model = DirtyModel()
    # model.build(input_shape=(None, 224, 224, 3))
    print(model.summary())