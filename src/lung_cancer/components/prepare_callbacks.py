import os
import tensorflow as tf
import time
from lung_cancer.entity.config_entity import PrepareCallbacksConfig


class PrepareCallback:
    def __init__(self, config: PrepareCallbacksConfig):
        self.config = config
        self.early_stopping = tf.keras.callbacks.EarlyStopping(monitor= 'val_loss', patience= 15)

    @property
    def _create_tb_callbacks(self):
        timestamp = time.strftime("%Y-%m-%d-%H-%M-%S")
        tb_running_log_dir = os.path.join(
            self.config.tensorboard_root_log_dir,
            f"tb_logs_at_{timestamp}",
        )
        return tf.keras.callbacks.TensorBoard(log_dir=tb_running_log_dir)
    

    @property
    def _create_ckpt_callbacks(self):
        return tf.keras.callbacks.ModelCheckpoint(
            filepath=str(self.config.checkpoint_model_filepath),
            save_best_only=True
        )
    

    def get_tb_ckpt_callbacks(self):
        return [
            self._create_tb_callbacks,
            self._create_ckpt_callbacks,
            self.early_stopping
        ]