from flask import Flask, request
from flask_cors import CORS, cross_origin
import tensorflow as tf
from werkzeug.utils import secure_filename
from werkzeug.datastructures import  FileStorage

#Import các thư viên
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model

from PIL import Image
import pickle
import numpy as np
import os
import cv2, base64
import datetime


sess = tf.compat.v1.Session()
graph = tf.compat.v1.get_default_graph()

with sess.as_default():
    with graph.as_default():
        #load model
        model = 5

# Viet ham tao model
def get_extract_model():
    vgg16_model = VGG16(weights = "imagenet")
    extract_model = Model(inputs=vgg16_model.inputs, outputs = vgg16_model.get_layer("fc1").output)
    return extract_model

def chuyen_base64_sang_anh(anh_base64): 
    try: 
        anh_base64 = np.fromstring(base64.b64decode(anh_base64), dtype=np.uint8)
        anh_base64 = cv2.imdecode(anh_base64, cv2.IMREAD_ANYCOLOR)
    except: 
        print("Return none")
        return None
    print("Return image")
    return anh_base64

# Ham chuyen doi hinh anh thanh tensor
def image_processing(img):
    img = img.resize((224,224))
    img = img.convert("RGB")
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x

def extract_vector(model, image_path):
    print("Xu ly: ", image_path)
    img = Image.open(image_path)
    img_tensor = image_processing(img)

    #Trich dac trung
    vector = model.predict(img_tensor)[0]
    #Chuan hoa vector = chia cho L2 norm (khoang cach vector hien tai toi goc toa do)
    vector = vector/np.linalg.norm(vector)
    return vector



#Load 4700 vector tu vector.pkl ra colection
vectors = pickle.load(open("vectors.pkl","rb"))
paths = pickle.load(open("paths.pkl","rb"))



#Ve len man hinh
import matplotlib.pyplot as plt
import math 
import json





app = Flask(__name__)

CORS(app)
app.config['CORS_HEADER'] = 'Content-Type'

@app.route('/', methods=['GET'])
def init_page_home():
    print("hello")
    return "Hello Thinh With Python"


@app.route('/init', methods=['GET'])
def init_page():
    print("hello")
    return "init"


@app.route('/searching', methods=['POST'])
@cross_origin(origin='*')
def add_process():
   # print("vao ham")
    base64Image = request.form.get('base64Image')
    #print("base64: ", base64Image)
    f = chuyen_base64_sang_anh(base64Image)

    currentDT = datetime.datetime.now()

    file_rename = "user_image/"+ currentDT.strftime("%Y%m%d%H%M%S") + "_xxx.jpg"

    #f.save(file_rename)

    cv2.imwrite(file_rename, f)

    search_image = file_rename
    model = get_extract_model()

    search_vector = extract_vector(model, search_image)
    #Tinh khoang cach tu vector search den tat cac cac vector trong kho

    distance = np.linalg.norm(vectors - search_vector, axis = 1)

    #Sap xep va lay k vector co khoang cach ngan nhat
    K =16
    ids = np.argsort(distance)[:K]
    
    #Tao output
    nearest_image = [(paths[id], distance[id]) for id in ids]
    axes = []
   # grid_size = int(math.sqrt(K))
   # fig = plt.figure(figsize=(50,50))
    #list=[]
    print(len(nearest_image))
    for id in range(K):
        draw_image = nearest_image[id]
        item = {
            "path":draw_image[0].replace("oxbuild_images-v1\\", "" ).replace(".jpg",""),
            "trust":float(round(draw_image[1], 2))
        }
        axes.append(item)
        #axes.append(fig.add_subplot(grid_size, grid_size, id+1))
        #axes[-1].set_title(draw_image[1])
       # plt.imshow(Image.open(draw_image[0]))

    #fig.tight_layout()
    #plt.show()
    #print(axes)
    return axes

if __name__ == '__main__':
    app.run(host='0.0.0.0')
