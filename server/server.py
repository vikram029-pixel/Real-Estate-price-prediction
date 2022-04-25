from flask import Flask,request, jsonify
import model
import json
from types import SimpleNamespace
from flask_cors import CORS, cross_origin

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

@app.route('/')
@cross_origin()
def hello_world():
   return 'House Price Prediction'

@app.route('/predictPrice', methods=['POST'])
@cross_origin()
def predictPrice():
    data = request.json
    print(data['sqft'])
    pricePredicted=model.predict_price(data['location'],data['sqft'],data['bath'],data['bhk']);
    return jsonify(pricePredicted)

if __name__ == '__main__':
   app.run()
