import tensorflow as tf
import keras
import argparse

def set_gpu_memory_growth(device_type="GPU"):
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        except RuntimeError as e:
            print(e)
    
def set_gpu_memory_limit(memory_limit = 8192, device_type="GPU"):
    gpus = tf.config.experimental.list_physical_devices(device_type)
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_virtual_device_configuration(gpu, [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=memory_limit)])
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # 메모리 제한 설정 실패할 경우 예외 처리
            print(e)

def parser_opt(): 
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", dest="epochs", type=int, default=30, help="total training epochs")
    parser.add_argument("--finetune_epochs", dest="finetune_epochs", type=int, default=10, help="total finetune epochs")
    parser.add_argument("--batch_size", dest="batch_size", type=int, default=32, help="total batch size")
    parser.add_argument("--broken_dir", dest="broken_dir", type=str, default="./dataset/datasets_broken/")
    parser.add_argument("--dirty_dir", dest="dirty_dir", type=str, default="./dataset/datasets_dirty2/")
    parser.add_argument("--save_dir", dest="save_dir", type=str, default="./outputs/")
    parser.add_argument("--training", dest="training", type=bool, default=True)
    # parser.add_argument("--training", dest="training", type=bool, default=False)
    return parser.parse_args()
    
def set_model_to_train(model, save_path, finetune=False):
    set_model_compile(model, finetune=finetune)
    return get_callbacks(save_path)

def get_callbacks(save_path):
    callbacks = []
    callbacks.append(tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3))
    callbacks.append(tf.keras.callbacks.ModelCheckpoint(filepath=save_path + "/checkpoints/", save_weights_only=True, monitor='val_accuracy', mode='max', save_best_only=True))
    return callbacks

def set_model_compile(model, finetune):
    if not finetune:
        model.compile(
            optimizer = keras.optimizers.Adam(),
            loss = keras.losses.CategoricalCrossentropy(),
            # metrics=[keras.metrics.CategoricalAccuracy()], 
            metrics=[keras.metrics.Precision(), keras.metrics.Recall(), keras.metrics.CategoricalAccuracy()],
        )
    else:
        # model.trainable = True
        model.compile(
            optimizer = keras.optimizers.Adam(1e-5),
            loss = keras.losses.CategoricalCrossentropy(),
            # metrics=[keras.metrics.CategoricalAccuracy()], 
            metrics=[keras.metrics.Precision(), keras.metrics.Recall(), keras.metrics.CategoricalAccuracy()],
        )

def save_model(model, save_path):
    model.save(save_path +"/"+ model.name + ".h5")
    

    