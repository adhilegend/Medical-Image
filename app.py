from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np

# Keras
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
import keras.preprocessing
from keras.layers import Conv2D
from keras.preprocessing import image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer
import tensorflow as tf

# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
MODEL_PATH = 'models/sequential.h5'
MODEL_PATH = 'models/cnn.h5'
#MODEL_PATH = 'models/Derma_inception_3.9.h5'
#MODEL_PATH = 'models/resnet18-model.h5'

# Load your trained model
model = load_model(MODEL_PATH)
model.make_predict_function()          # Necessary
# print('Model loaded. Start serving...')

# You can also use pretrained model from Keras
# Check https://keras.io/applications/
#from keras.applications.resnet50 import ResNet50
#model = ResNet50(weights='imagenet')
#model.save('')
print('Model loaded. Check http://127.0.0.1:5000/')


def model_predict(image_path, model):
    img_rows = 28
    img_cols = 28
    image = tf.keras.preprocessing.image.load_img(image_path, target_size=(img_rows, img_cols))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr]) 
    input_arr = input_arr.astype('float32') / 255.  # This is VERY important

 # Convert single image to a batch.
    x = np.random.randint(0,10,(28,28,3))
    x = np.expand_dims(x, axis=0)
   # img = image.load_img(img_path, target_size=(28, 28)) 
    # Preprocessing the image
    #x = image.img_to_array(img)
    #print("Value is %s" % (x.ndim))
    #x = np.true_divide(x, 255)
    #x = np.expand_dims(input_arr, axis=(0,1))
    #print(x)
    # Be careful how your trained model deals with the input
    # otherwise, it won't make correct prediction!
    #x = preprocess_input(x, mode='caffe')
    #convarr = Conv2D(32,kernel_size=(3,3),#activation='relu',input_shape=input_arr.shape)
    preds = model.predict(x)
    return preds


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path, model)
        print(preds)
        # Process your result for human
        pred_class = preds.argmax(axis=-1)            # Simple argmax
        #pred_class = decode_predictions(preds, top=5) 
        print(pred_class)
          # ImageNet Decode
    
        result = str(pred_class)               # Convert to string
        return result
    return None


if __name__ == '__main__':
    app.run(debug=True)