from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np
from PIL import Image
#import keras

# Keras
from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
#MODEL_PATH ='C:/Users/Abhishek/Downloads/Malaria-Detection-master/Malaria-Detection-master/model_vgg19.h5'

# Load your trained model
model = load_model('model_vgg19.h5')
#model=keras.models.load_model('model_vgg19.h5')
def tumor(number):
    
    if number == 0:
        
        return "Not a tumor"
    
    else:
        
        return "a tumor"





def model_predict(img_path, model):
    img = Image.open(img_path)
    
    #img = Image.open('/content/drive/MyDrive/computer vision/archive (4)/testing/brain_tumor_dataset/yes/Y100.JPG')

    x = np.array(img.resize((64,64)))

    x = x.reshape(1, 64, 64, 3)

    result = model.predict([x])

    classification = np.where(result == np.amax(result))[1][0]

#
    preds=(str(result[0][classification]*100) + '% Confidence This Is ' + tumor(classification))

   # plt.imshow(img)

    # Preprocessing the image
    #x = image.img_to_array(img)
    #x = np.true_divide(x, 255)
    ## Scaling
    #x=x/255
    #x = np.expand_dims(x, axis=0)
   

    # Be careful how your trained model deals with the input
    # otherwise, it won't make correct prediction!
    #x = preprocess_input(x)

    #preds = model.predict(x)
    #preds=np.argmax(preds, axis=1)
    #if preds==0:
      #  preds="The Person is Infected With Pneumonia"
    #else:
     #   preds="The Person is not Infected With Pneumonia"
    
    
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
        result=preds
        return result
    return None


if __name__ == '__main__':
    app.run(debug=True)
