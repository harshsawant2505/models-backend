from flask import Flask, request, jsonify
from models.image import cosine_calculate
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import base64
app = Flask(__name__)
img='th (1).jpg'

# Load places data
places = pd.read_csv('./models/places.csv')
nb_places = len(places)

# Define SAE class
class SAE(nn.Module):
    def __init__(self):
        super(SAE, self).__init__()
        self.fc1 = nn.Linear(nb_places, 20)
        self.fc2 = nn.Linear(20, 10)
        self.fc3 = nn.Linear(10, 20)
        self.fc4 = nn.Linear(20, nb_places)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        x = self.fc4(x)
        return x

# Load your model
sae = SAE()
sae.load_state_dict(torch.load('./models/sae_model.pth'))
sae.eval()

def recommend_places(sae, user_reviews, nb_places):
    user_reviews = torch.FloatTensor(user_reviews)
    input_data = user_reviews.unsqueeze(0)
    with torch.no_grad():
        predicted_ratings = sae(input_data)
    predicted_ratings = predicted_ratings.numpy().flatten()
    user_ratings = user_reviews.numpy()
    unrated_places = np.where(user_ratings == 0)[0]
    recommended_place_indices = unrated_places[np.argsort(predicted_ratings[unrated_places])[::-1]]
    recommended_place_names = places.iloc[recommended_place_indices[:5]]['place_name'].values
    recommended_place_ratings = predicted_ratings[recommended_place_indices][:5]
    return str(list(zip(recommended_place_names, recommended_place_ratings)))


def decode_base64_to_image(encoded_string, output_path):
    img_data = base64.b64decode(encoded_string)
    with open(output_path, 'wb') as output_file:
        output_file.write(img_data)



decode_path_user = 'models/user_image.jpg'
decode_path_compare = 'models/compare_image.jpg'

@app.route("/", methods=["POST"])

def home():
    

    data = request.get_json()
    image1 = data.get('image1')
    image2 = data.get('image2')
    print(image1)
    print(image2)

    decode_path1 = 'models\image1.jpg'
    decode_path2 = 'models\image2.jpg'

    decode_base64_to_image(image1, decode_path1)
    decode_base64_to_image(image2, decode_path2)
    
     
    return  str(cosine_calculate(decode_path1 ,  decode_path2)) # This should display on the webpage

@app.route('/calculate_similarity', methods=['GET'])
def calculate_similarity_route():
    from models.imageclassification import calculate_similarity
    user_image = decode_path_user
    compare_image = request.args.get(decode_path_compare)  # Default to a predefined image
    data = request.get_json()
    image1 = data.get('image1')
    decode_path1 = 'models\image1.jpg'
    decode_base64_to_image(image1, decode_path1)
    similarity_score = calculate_similarity(decode_path1)
    return jsonify({"Image similarity score": similarity_score})

@app.route('/recommend', methods=['GET'])
def recommend():
    user_ratings = [5, 0, 3, 3, 4, 0, 2, 5, 1, 0, 3, 4, 5, 1, 2, 3, 4, 2, 5, 0, 1, 3, 4, 0, 2, 5, 1, 4, 3, 5, 0, 2, 4, 3, 1, 5, 2, 4, 0, 3, 4, 2, 1, 5, 0, 3, 4, 2, 5, 1, 3, 4, 0, 2, 5, 3, 1, 4, 2, 0, 5, 3, 4, 1, 2, 0, 5, 3, 1, 4, 2, 0, 5, 3, 1, 4, 0, 2, 5, 3, 1, 4, 2, 0, 5, 3, 4, 2, 1, 0, 5, 3, 1, 4, 2, 0, 5, 3, 1, 4]
    recommended_places = recommend_places(sae, user_ratings, nb_places)
    return jsonify(recommended_places)


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8000, debug=True)
