import os
import numpy as np
from PIL import Image
import cv2
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
import warnings
from PIL import Image, ImageEnhance
warnings.filterwarnings('ignore')
import tensorflow as tf
from keras.models import load_model
from keras.applications.vgg16 import preprocess_input
import numpy as np
from keras.preprocessing import image



app = Flask(__name__)

print('Model loaded. Check http://127.0.0.1:5000/')


def get_className(classNo):
	if classNo==0:
		return "Normal"
	elif classNo==1:
		return "Pneumonia"


def getResult(img):
    model=load_model('chest_xray.h5') 
    img_file=image.load_img(img,target_size=(224,224))
    x=image.img_to_array(img_file)
    x=np.expand_dims(x, axis=0)
    img_data=preprocess_input(x)
    classes=model.predict(img_data)
    global result
    result=classes
    if result[0][0]>0.5:
            return "Normal"
    else:
            return "Pneumonia"


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        f = request.files['file']

        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        value=getResult(file_path)
        result=get_className(value) 
        return value
    return None


if __name__ == '__main__':
    app.run(debug=True)