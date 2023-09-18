from train import set_gpu_memory_growth, parser_opt, set_model_to_train, save_model
from dataset import get_broken_labels, get_dirty_labels, get_merge_datas
from dataset import DataGenerator
from sklearn.model_selection import train_test_split
from model import get_bd_model
from utils import make_dir, set_save_path

def main(opt):
    set_gpu_memory_growth()
    
    broken_labels = get_broken_labels(opt.broken_dir)
    dirty_labels = get_dirty_labels(opt.dirty_dir)
    total_image_list, total_broken_labels, total_dirty_labels = get_merge_datas(broken_labels, dirty_labels)
    
    train_X, test_x, train_Y, test_y = train_test_split(total_image_list, total_dirty_labels, test_size=2000./len(total_image_list), random_state=42)
    train_x, valid_x, train_y, valid_y = train_test_split(train_X, train_Y, test_size = 0.1, random_state=42)
    
    
    
    # # # 모델 세팅, 클래스로 변경할 필요 있음.
    # model = get_bd_model()
    import tensorflow as tf
    from tensorflow.keras.applications import efficientnet_v2, EfficientNetV2B0 
    num_classes=7
    function="softmax"
    freeze=True
    
    base_model = EfficientNetV2B0(weights='imagenet')
    feature_extractor = tf.keras.Model(inputs=base_model.input, outputs=base_model.get_layer('top_activation').output, name=base_model.name+"_FE")
    feature_extractor.trainable = False
    
    dx1 = tf.keras.layers.Conv2D(64, (3,3), activation="relu")(feature_extractor.output)
    dx2 = tf.keras.layers.MaxPooling2D()(dx1)
    dx3 = tf.keras.layers.Flatten()(dx2)
    dx4 = tf.keras.layers.Dense(128, activation='relu')(dx3)
    dd1 = tf.keras.layers.Dropout(rate=0.2)(dx4)
    
    total_output = tf.keras.layers.Dense(7, activation=function, name="output")(dd1)
    
    model = tf.keras.Model(feature_extractor.input, total_output, name=base_model.name+"_db_"+function)
    
    
    train_datas = DataGenerator(train_x, train_y, opt.batch_size, model.input_shape)
    valid_datas = DataGenerator(valid_x, valid_y, opt.batch_size, model.input_shape)
    test_datas = DataGenerator(test_x, test_y, opt.batch_size, model.input_shape)
    
    
    # # # 메인 훈련
    save_path = set_save_path(opt.save_dir + model.name)
    callbacks = set_model_to_train(model, save_path)
    model.fit(train_datas, epochs=opt.epochs, validation_data = valid_datas, callbacks=callbacks)
    
    # # # 파인 튜닝
    model.trainable = True
    
    import keras
    base_model.compile(
        optimizer=keras.optimizers.Adam(1e-5),
        loss={'output_dirty':keras.losses.CategoricalCrossentropy()},
        metrics={'output_dirty':keras.metrics.CategoricalAccuracy()}, 
    )
    # callbacks = set_model_to_train(model, save_path, finetune=True)
    model.fit(train_datas, epochs=opt.finetune_epochs, validation_data=valid_datas, callbacks=callbacks)

    # # # 모델 저장
    save_model(model, save_path)
    
if __name__ == "__main__":
    opt = parser_opt()
    main(opt)