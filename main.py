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
    total_image_list, total_dirty_labels = get_merge_datas(broken_labels, dirty_labels)
    
    train_X, test_x, train_Y, test_y = train_test_split(total_image_list, total_dirty_labels, test_size=2000./len(total_image_list), random_state=42)
    train_x, valid_x, train_y, valid_y = train_test_split(train_X, train_Y, test_size = 0.1, random_state=42)
    
    if opt.training == True:
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
        
        dx1 = tf.keras.layers.Conv2D(64, (3,3), activation="relu", padding='same')(feature_extractor.output)
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
        
        model.evaluate(test_datas)
        # # # 파인 튜닝
        model.trainable = True
        
        import keras
        base_model.compile(
            optimizer=keras.optimizers.Adam(1e-5),
            loss={'output_dirty':keras.losses.CategoricalCrossentropy()},
            metrics=[keras.metrics.Precision(), keras.metrics.Recall(), keras.metrics.CategoricalAccuracy()],
        )
        # callbacks = set_model_to_train(model, save_path, finetune=True)
        model.fit(train_datas, epochs=opt.finetune_epochs, validation_data=valid_datas, callbacks=callbacks)
        
        model.evaluate(test_datas)

        # # # 모델 저장
        save_model(model, save_path)
    else:
        # # 테스트 결과 확인.
        print("test")
        
        
        if(0): # # # 왜 DataGenerator의 predict가 재대로 수행되지 않는지 확인해야 함.
            load_path = "./outputs/efficientnetv2-b0_db_softmax_e30+f10/efficientnetv2-b0_db_softmax.h5"
            import tensorflow as tf
            import keras
            import numpy as np
            model = keras.models.load_model(load_path)
            input_shape = model.input_shape[1:]
            input_size = input_shape[:-1]

            dummy = np.zeros(input_shape)
            dummy = np.expand_dims(dummy, axis=0)
            model.predict(dummy)
            
            train_datas = DataGenerator(train_x, train_y, 1, model.input_shape)
            valid_datas = DataGenerator(valid_x, valid_y, 1, model.input_shape)
            test_datas = DataGenerator(test_x, test_y, 1, model.input_shape)
            
            pred_train = model.predict(train_datas)
            pred_valid = model.predict(valid_datas)
            pred_test = model.predict(test_datas)
            
            train_result = []
            valid_result = []
            test_result =[]
            for i in [None, "micro", "macro", "weighted"]:
                metric_train = keras.metrics.F1Score(name='f1_score', threshold=None, average=i)
                metric_valid = keras.metrics.F1Score(name='f1_score', threshold=None, average=i)
                metric_test  = keras.metrics.F1Score(name='f1_score', threshold=None, average=i)
                
                metric_train.update_state(train_y, pred_train)
                metric_valid.update_state(valid_y, pred_valid)
                metric_test.update_state(test_y, pred_test)
                
                train_result.append(metric_train.result())
                valid_result.append(metric_valid.result())
                test_result.append(metric_test.result())
                
                metric_train.reset_state()
                metric_valid.reset_state()
                metric_test.reset_state()
                
            model.evaluate(test_datas)

            model.compile(metrics= [keras.metrics.Precision(name="precision"), 
                                    keras.metrics.Recall(name="recall"), 
                                    keras.metrics.TruePositives(name='true_positive'),
                                    keras.metrics.TrueNegatives(name='true_negative'),
                                    keras.metrics.FalsePositives(name='false_positive'),
                                    keras.metrics.FalseNegatives(name='false_negative'),
                                    ])
                    
        else:
            from natsort import natsorted
            import numpy as np
            import os
            import keras
            import cv2
            
            img_list = test_x
            img_list = natsorted(img_list)
            
            lbl_list = [name.replace(".jpg", ".txt") for name in img_list]
            dirty_lbl_list = np.array(lbl_list)
            _, labels_eigen_vector = np.linalg.eig(np.diag((1,2,3,4,5,6,7)))
            
            actual = []
            for idx, lbl_name in enumerate(dirty_lbl_list):
                if os.path.exists(lbl_name):
                    dirty_label = int(open(lbl_name, 'r').readlines()[0])
                else:
                    dirty_label = -1
                # import pdb; pdb.set_trace()
                onehot = labels_eigen_vector[dirty_label+1]
                actual.append(onehot)
            actual = np.array(actual)

            model = keras.models.load_model("./outputs/efficientnetv2-b0_db_softmax_e30+f10/efficientnetv2-b0_db_softmax.h5")
            # model = keras.models.load_model("./outputs/efficientnetv2-b0_db_softmax_02/efficientnetv2-b0_db_softmax.h5")
            
            input_shape = model.input_shape[1:]
            input_size = input_shape[:-1]

            dummy = np.zeros(input_shape)
            dummy = np.expand_dims(dummy, axis=0)
            model.predict(dummy)

            imgs = np.array([cv2.resize(cv2.imread(name), input_size) for name in img_list])
            preds = model.predict(imgs, verbose=0)

            mPrecision = keras.metrics.Precision()
            mRecall = keras.metrics.Recall()
            mF1None = keras.metrics.F1Score(average=None)
            mF1Micro = keras.metrics.F1Score(average='micro')
            mF1Macro = keras.metrics.F1Score(average='macro')
            mF1Weighted = keras.metrics.F1Score(average="weighted")

            mPrecision.update_state(actual, preds)
            mRecall.update_state(actual, preds)
            mF1None.update_state(actual, preds)
            mF1Micro.update_state(actual, preds)
            mF1Macro.update_state(actual, preds)
            mF1Weighted.update_state(actual, preds)

            print("Precision : ", mPrecision.result())
            print("Recall    : ", mRecall.result())
            print("F1 None   : ", mF1None.result())
            print("F1 Micro  : ", mF1Micro.result())
            print("F1 Macro  : ", mF1Macro.result())
            print("f1 weight : ", mF1Weighted.result())

            print("check Result")
            

        
    
if __name__ == "__main__":
    opt = parser_opt()
    main(opt)