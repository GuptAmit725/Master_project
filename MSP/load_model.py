import tensorflow as tf

class get_model:
    def __init__(self):
        pass

    def get_classifier(self,class_PATH):
        classifier = tf.keras.models.load_model(class_PATH)
        return classifier

    def get_gradModel(self, grad_PATH):
        grad_model = tf.keras.models.load_model(grad_PATH)
        return grad_model