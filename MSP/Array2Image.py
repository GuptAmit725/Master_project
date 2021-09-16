import tensorflow as tf
from logger import logs

log_it = logs()
log_it.log(f"=====Into the Array2Image.py Module=====")

class array_to_image:
    def __init__(self, img_array):
        self.img = img_array

    def get_image_from_array(self):
        log_it.log(f"=====Converting array into  image=====")
        img = tf.keras.utils.array_to_img(self.img)
        log_it.log(f"=====Converting array into  image : successful=====")
        return img


