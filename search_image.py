#Import các thư viên
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model

from PIL import Image
import pickle
import numpy as np
import os

# Viet ham tao model
def get_extract_model():
    vgg16_model = VGG16(weights = "imagenet")
    extract_model = Model(inputs=vgg16_model.inputs, outputs = vgg16_model.get_layer("fc1").output)
    return extract_model

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

search_image = "dataset/4511.jpg"

model = get_extract_model()

search_vector = extract_vector(model, search_image)

#Load 4700 vector tu vector.pkl ra colection
vectors = pickle.load(open("vectors.pkl","rb"))
paths = pickle.load(open("paths.pkl","rb"))

#Tinh khoang cach tu vector search den tat cac cac vector trong kho

distance = np.linalg.norm(vectors - search_vector, axis = 1)

#Sap xep va lay k vector co khoang cach ngan nhat
K =16
ids = np.argsort(distance)[:K]

#Tao output
nearest_image = [(paths[id], distance[id]) for id in ids]

#Ve len man hinh
import matplotlib.pyplot as plt
import math 
axes = []
grid_size = int(math.sqrt(K))
fig = plt.figure(figsize=(50,50))

for id in range(K):
    draw_image = nearest_image[id]
    axes.append(fig.add_subplot(grid_size, grid_size, id+1))
    axes[-1].set_title(draw_image[1])
    plt.imshow(Image.open(draw_image[0]))

fig.tight_layout()
plt.show()






