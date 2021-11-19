import sys

import tflite_runtime.interpreter as tflite
import numpy as np

from PIL import Image
from skimage import transform

filepath = sys.argv[1];
print("File Path: ", filepath)

def load(filename):
    np_image = Image.open(filename)
    np_image = np.array(np_image).astype('float32')/255
    np_image = transform.resize(np_image, (28, 28, 1))
    np_image = np.expand_dims(np_image, axis=0)
    return np_image

image = load(filepath)

interpreter = tflite.Interpreter(model_path="./model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

interpreter.set_tensor(input_details[0]['index'], image)

interpreter.invoke()

output_data = interpreter.get_tensor(output_details[0]['index'])
print("Model Prediction: ", np.argmax(output_data))
print("Model Confidence: {:02.2f}%".format(np.max(output_data)*100))

    #num = input("Choose number to predict: ")
    #num = int(num)
