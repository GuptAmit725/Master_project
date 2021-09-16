import tensorflow as tf
from classification import prediction
from load_model import get_model
from Explaination import explain

class_path = 'C:/Users/amiti/Desktop/Master_project/MSP/static/inceptionV3_trained.h5'
grad_path = 'C:/Users/amiti/Desktop/Master_project/MSP/static/grad_model2.h5'

GET_MODEL = get_model()
classifer_model = GET_MODEL.get_classifier(class_path)
grad_model = GET_MODEL.get_gradModel(grad_path)

class output:
    def __init__(self,img_path):
        self.img = img_path

    def get_class(self):
        img_1 = tf.keras.preprocessing.image.load_img(self.img, target_size=(128, 128, 3))
        img_1 = tf.keras.preprocessing.image.img_to_array(img_1).reshape(1,128,128,3)
        img_1 = img_1 / 255.

        print(img_1.shape)

        return classifer_model.predict(img_1).argmax(), classifer_model.predict(img_1).max()

    def ExplainOutput(self, prediction):
        if prediction:
            explaining = explain(grad_model, classifer_model)

            return explaining.show_lobe(self.img), explaining.show_pixels(self.img)
        else:
            return f"There is nothing to explain as the X-Ray is NORMAL", ''

