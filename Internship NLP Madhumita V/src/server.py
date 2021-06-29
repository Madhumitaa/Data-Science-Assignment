from nc_extraction import read_json
from flask import Flask, render_template, request
import pickle
import numpy as np


model = pickle.load(open('top_nc_model.pkl', 'rb'))

app = Flask(__name__)

from flask import Flask
from flask import request
  
app = Flask(__name__)
  
@app.route('/postjson', methods = ['POST'])
def postJsonHandler():
    print (request.is_json)
    content = request.get_json()
    return content
  
app.run(host='0.0.0.0', port= 8090)

if __name__ == '__main__':
    app.run(debug=True)