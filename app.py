
from config import app
from flask import request, jsonify


@app.route("/", methods=["GET"])
def home():
    return jsonify({"data": "Data from python flask server get route"})




if __name__ == "__main__":
    app.run(host = 'localhost', port=8000, debug=True)