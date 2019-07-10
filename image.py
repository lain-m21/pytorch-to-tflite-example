from PIL import Image
import numpy as np


MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def load_and_preprocess_image(path, image_size=(224, 224)):
    image = Image.open(path)
    image = image.convert('RGB')
    image = image.resize(image_size)
    image_array = np.asarray(image)
    image_array = image_array.transpose((2, 0, 1))  # H x W x C --> C x H x W
    image_array = preprocess_input(image_array, MEAN, STD)
    return np.expand_dims(image_array, 0)  # C x H x W --> N x C x H x W


def preprocess_input(array, mean, std):
    array = array.astype(np.float32)
    array /= 255
    array[0, :, :] = (array[0, :, :] - mean[0]) / std[0]
    array[1, :, :] = (array[1, :, :] - mean[1]) / std[1]
    array[2, :, :] = (array[2, :, :] - mean[2]) / std[2]
    return array


