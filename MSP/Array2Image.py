import tensorflow as tf

class array_to_image:
    def __init__(self, img_array):
        self.img = img_array

    def get_image_from_array(self):
        return tf.keras.utils.array_to_img(self.img)

