# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 22:37:37 2019

@author: SJana
"""
from flask import Flask, request, jsonify

import traceback

from keras.models import load_model
from keras.preprocessing.image import load_img,img_to_array
import numpy as np



# Your API definition
app = Flask(__name__)
@app.route('/predict', methods=['POST'])
def predict():
    if classifier:
        try:
            path = request.json
            print(path['Path'])
            image_path = path['Path']
            test_img = load_img(image_path,target_size=(64,64))
            test_img.size
            test_image =img_to_array(test_img)
            test_image=np.expand_dims(test_image,axis=0)
            test_image.shape
            predicted_label=classifier.predict_proba(test_image)
            for i in predicted_label:
                    image_label=i
            if image_label[0]==1:
                    label="Aadhar Card"
            elif image_label[0]==0:
                    label="Voter ID"
            return jsonify({'Predicted_label': label})
           
        except:

            return jsonify({'trace': traceback.format_exc()})
    else:
        print ('Train the model first')
        return ('No model here to use')

if __name__ == '__main__':
    try:
        port = int(sys.argv[1]) # This is for a command-line input
    except:
        port = 8085 # If you don't provide any port the port will be set to 12345

    classifier = load_model('document_classification_model.h5') # Load "model.h5"
    print ('Model loaded')
    app.run(port=port, debug=True)






















