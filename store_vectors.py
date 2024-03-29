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



# Dinh nghi thu muc data
data_folder = "oxbuild_images-v1"
# Khoi tao model
model = get_extract_model()

vectors = []
paths = []
for image_path in os.listdir(data_folder):
    #get full path 
    image_path_full = os.path.join(data_folder, image_path)
    #Trich dac trung
    image_vector = extract_vector(model, image_path_full)

    vectors.append(image_vector)
    paths.append(image_path_full)

#save vao file
vector_file = "vectors.pkl"
path_file = "paths.pkl"

pickle.dump(vectors, open(vector_file, "wb"))
pickle.dump(paths, open(path_file, "wb"))


