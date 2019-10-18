# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 22:37:37 2019

@author: SJana
"""

from keras.models import load_model
from keras.preprocessing.image import load_img,img_to_array
import numpy as np

model = load_model('document_classification_model.h5')

test_img = load_img('Documents/test_img/test/test1.jpg',target_size=(64,64))
test_img.size
test_img.show()
test_image =img_to_array(test_img)
test_image=np.expand_dims(test_image,axis=0)
test_image.shape
image_label=model.predict(test_image)

print(image_label)
