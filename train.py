import tensorflow as tf
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
    parser.add_argument("--epochs", dest="epochs", type=int, default=50, help="total training epochs")
    parser.add_argument("--finetune_epochs", dest="finetune_epochs", type=int, default=10, help="total finetune epochs")
    parser.add_argument("--batch_size", dest="batch_size", type=int, default=32, help="total batch size")
    parser.add_argument("--broken_dir", dest="broken_dir", type=str, default="./dataset/datasets_broken/")
    parser.add_argument("--dirty_dir", dest="dirty_dir", type=str, default="./dataset/datasets_dirty2/")
    parser.add_argument("--save_dir", dest="save_dir", type=str, default="./outputs/")
    return parser.parse_args()
    
def get_save_path():
    return "save_path/"
    
class Trainer:
    def __init__(self, model, epochs, batch, loss_fn, optimizer):
        self.model = model
        self.epochs = epochs
        self.batch = batch
        self.loss_fn = loss_fn

    def train(self, train_dataset, train_metric):
        for epoch in range(self.epochs):
            print("\nStart of epoch %d"%(epoch,))
            for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
                with tf.GradientTape() as tape:
                    logits = self.model(x_batch_train)
                    loss_value = self.loss_fn(y_batch_train, logits)
                grads = tape.gradient(loss_value, self.model.trainable_weights)
                self.optimizer.apply_gradients(zip(grads, self.model.trainable_weight))

                train_metric.update_state(y_batch_train, logits)

                if step % 5 == 0:
                    print(
                        "Training loss (for one batch) at step %d: %.4f"
                        % (step, float(loss_value))
                    )
                    print("Seen so far: %d samples" % ((step + 1) * self.batch))
                    print(train_metric.result().numpy())

            train_acc = train_acc_metric.result()
            print("Training acc over epoch: %.4f" % (float(train_acc),))
