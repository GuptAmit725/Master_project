import tensorflow as tf
import numpy as np
import find_lobe

class explain:

    def __init__(self, grad_model, classifier):
        self.grad_model = grad_model
        self.model = classifier

    def show_pixels(self,img_path):
        img = img_1 = tf.keras.preprocessing.image.load_img(img_path)
        img_1 = tf.keras.preprocessing.image.load_img(img_path,target_size=(128,128,3))
        img_1 = tf.keras.preprocessing.image.img_to_array( img_1)
        img_1 = img_1/255.

        grad = self.grad_model.layers[1].get_weights()[0].reshape(128,128,3)
        grad = (grad - grad.min()) / (grad.max() - grad.min())
        # For lungs segmentation
        grad = img_1 * grad
        grad = img_1 + grad
        grad = (grad - grad.min()) / (grad.max() - grad.min())
        #for activation points, which contributes more than 70% in the classification
        grad[grad < 0.5] = 0

        return grad

    def show_lobe(self, img_path):
        # Loading the image for testing.
        img_r = tf.keras.preprocessing.image.load_img(img_path, target_size=(128, 128, 3))
        img_r = tf.keras.preprocessing.image.img_to_array(img_r)
        img_r = img_r / 255.

        img = find_lobe.find_correct_lobe(img_r)

        return img

