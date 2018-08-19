
from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
from keras.models import Model
import numpy as np

class Img2Vec():

    def __init__(self):

        model = ResNet50(weights='imagenet')

        layer_name = 'avg_pool'
        self.intermediate_layer_model = Model(inputs=model.input, outputs=model.get_layer(layer_name).output)


    def get_vec(self, image_path):
        """ Gets a vector embedding from an image
        :param image_path: path to image on filesystem
        :returns: numpy ndarray
        """

        img = image.load_img(image_path, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        intermediate_output = self.intermediate_layer_model.predict(x)

        return intermediate_output[0][0][0]