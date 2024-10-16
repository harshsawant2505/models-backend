from models.image import cosine_calculate
from config import app
from flask import request, jsonify

#done some


img = 'C:\Users\PARSHURAM\OneDrive\Desktop\models-backend\models\taj mahal3.jpg'

@app.route("/", methods=["GET"])
def home():
    k = cosine_calculate(img,img)
    return jsonify({"data": "Data from python flask server get route"})




if __name__ == "__main__":
    app.run(host = 'localhost', port=8000, debug=True)