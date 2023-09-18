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
    
    model = get_bd_model()
    
    train_datas = DataGenerator(train_x, train_y, opt.batch_size, model.input_shape)
    valid_datas = DataGenerator(valid_x, valid_y, opt.batch_size, model.input_shape)
    test_datas = DataGenerator(test_x, test_y, opt.batch_size, model.input_shape)
    
    save_path = set_save_path(opt.save_dir + model.name)
    
    callbacks = set_model_to_train(model, save_path)
    model.fit(train_datas, epochs=opt.epochs, validation_data = valid_datas, callbacks=callbacks)
    
    callbacks = set_model_to_train(model, save_path, finetune=True)
    model.fit(train_datas, epochs=opt.finetune_epochs, validation_data=valid_datas, callbacks=callbacks)

    save_model(model, save_path)
    
if __name__ == "__main__":
    opt = parser_opt()
    main(opt)