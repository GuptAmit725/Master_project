import tensorflow as tf
import find_lobe
from logger import logs

log_it = logs()
log_it.log(f"=====Into the Explanation Module=====")

class explain:

    def __init__(self, grad_model, classifier):
        self.grad_model = grad_model
        self.model = classifier

    def show_pixels(self,img_path):
        log_it.log(f"=====Into the Explanation Module and in show_pixels method  of explain class=====")
        img_1 = tf.keras.preprocessing.image.load_img(img_path,target_size=(128,128,3))
        img_1 = tf.keras.preprocessing.image.img_to_array( img_1)
        img_1 = img_1/255.
        log_it.log(f"=====Image preprocessing done.=====")

        grad = self.grad_model.layers[1].get_weights()[0].reshape(128,128,3)
        grad = (grad - grad.min()) / (grad.max() - grad.min())
        # For lungs segmentation
        log_it.log(f"=====Getting the lung segmentation=====")
        grad = img_1 * grad
        grad = img_1 + grad
        grad = (grad - grad.min()) / (grad.max() - grad.min())
        #for activation points, which contributes more than 70% in the classification
        log_it.log(f"=====Getting pixels value with greater than 0.7=====")
        grad[grad < 0.7] = 0

        return grad

    def show_lobe(self, img_path):
        # Loading the image for testing.
        log_it.log(f"=====Into the Explanation Module and in show_lobe method  of explain class=====")
        img_r = tf.keras.preprocessing.image.load_img(img_path, target_size=(128, 128, 3))
        img_r = tf.keras.preprocessing.image.img_to_array(img_r)
        img_r = img_r / 255.
        log_it.log(f"=====Image preprocessing done.=====")

        img = find_lobe.find_correct_lobe(img_r)
        log_it.log(f"=====Lobe found..=====")

        return img

