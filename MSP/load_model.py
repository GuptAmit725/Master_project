import tensorflow as tf
from logger import logs

log_it = logs()
log_it.log(f"=====Into the load_model Module=====")

class get_model:
    def __init__(self):
        pass

    def get_classifier(self,class_PATH):
        log_it.log(f"=====Loading classifier=====")
        classifier = tf.keras.models.load_model(class_PATH)
        log_it.log(f"=====classifier loading : successful=====")
        return classifier

    def get_gradModel(self, grad_PATH):
        log_it.log(f"=====Loading grad_model=====")
        grad_model = tf.keras.models.load_model(grad_PATH)
        log_it.log(f"=====grad_model loading : successful=====")
        return grad_model