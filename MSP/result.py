import tensorflow as tf
from load_model import get_model
from Explaination import explain
from logger import logs

log_it = logs()
log_it.log(f"=====Into the result.py module=====")

log_it.log(f"=====Declaring model paths.=====")
class_path = 'C:/Users/amiti/Desktop/Master_project/MSP/static/grad_model2.h5'
grad_path = 'C:/Users/amiti/Desktop/Master_project/MSP/static/grad_model2.h5'

log_it.log(f"=====Loading the model=====")
GET_MODEL = get_model()
classifer_model = GET_MODEL.get_classifier(class_path)
log_it.log(f"=====Classifier loaded=====")
grad_model = GET_MODEL.get_gradModel(grad_path)
log_it.log(f"=====Gradient explaining model loaded=====")

class output:
    def __init__(self,img_path):
        self.img = img_path

    def get_class(self):
        log_it.log(f"=====Getting the class predicition=====")
        img_1 = tf.keras.preprocessing.image.load_img(self.img, target_size=(128, 128, 3))
        img_1 = tf.keras.preprocessing.image.img_to_array(img_1).reshape(1,128,128,3)
        img_1 = img_1 / 255.
        log_it.log(f"=====input image resized and preprocessed for feeding into model=====")
        #print(img_1.shape)

        return classifer_model.predict(img_1).argmax(), classifer_model.predict(img_1).max()

    def ExplainOutput(self, prediction):
        log_it.log(f"=====Into the ExplainOutput function in result.py module=====")
        if prediction:
            explaining = explain(grad_model, classifer_model)
            log_it.log(f"=====prediction is 1 and making the instance of explaining class=====")
            return explaining.show_lobe(self.img), explaining.show_pixels(self.img)
        else:
            log_it.log(f"=====prediction is 0, explanation not needed. =====")
            return f"There is nothing to explain as the X-Ray is NORMAL", ''

