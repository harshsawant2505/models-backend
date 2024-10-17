from flask import Flask, request, jsonify
from models.imageclassification import calculate_similarity
from models.imagematch import cosine_calculate
import json
app = Flask(__name__)



import base64
def decode_base64_to_image(encoded_string, output_path):
    img_data = base64.b64decode(encoded_string)
    with open(output_path, 'wb') as output_file:
        output_file.write(img_data)


 



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

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8000, debug=True)
