import os
import numpy as np
import pandas as pd
import json  
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input
from scipy.spatial.distance import cosine

loaded_model = tf.keras.models.load_model('models\\vgg16_model.h5')




def extract_feature_vector(img_path):
    img = load_img(img_path, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    features = loaded_model.predict(img_array)
    print(features.flatten().shape)
    return features.flatten()


def calculate_similarity(new_img_path):
    with open('models\\class_features.json', 'r') as json_file:
      loaded_dict = json.load(json_file)

    new_features = extract_feature_vector(new_img_path)
    similarities = {}

    for class_name, data in loaded_dict.items():
        
        avg_features = np.mean(np.array(data['features']), axis=0)  
        similarity = 1 - cosine(new_features, avg_features)
        similarities[class_name] = similarity


    similarity_list = []  
    class_names = [] 

    for class_name, similarity in similarities.items():
      similarity_list.append(similarity)
      class_names.append(class_name)


    index = np.argmax(similarity_list)
    
    return class_names[index]
   
   


