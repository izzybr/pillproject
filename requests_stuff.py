import numpy as np
import cv2
import requests
from sliding_hog import extract_im, slideExtract, location_to_df


#r = requests.get('https://data.lhncbc.nlm.nih.gov/public/Pills/PillProjectDisc1/images/!0KGOH213DNYLLF5BOG6!L6-AFX1UD.JPG')
#mg = np.asarray(bytearray(r.content), dtype='uint8')
#img = cv2.imdecode(img, cv2.IMREAD_COLOR)

https://www.dropbox.com/s/bb6i5w4upzjyioi/svm_prob_clf_pkl2





r = requests.get('https://data.lhncbc.nlm.nih.gov/public/Pills/PillProjectDisc1/images/!Z_!JN0!T6ULVSWNTYYFROROW88XSQ.JPG')

from urllib.request import urlopen
from joblib import load
from sklearn import svm
import pickle
import numpy as np
import cv2
import pandas as pd
from skimage.feature import hog

def load_model():
    return load(urlopen('https://www.dropbox.com/s/bb6i5w4upzjyioi/svm_prob_clf_pkl2?dl=1'))

def get_image():
    r = requests.get('https://data.lhncbc.nlm.nih.gov/public/Pills/PillProjectDisc1/images/!0KGOH213DNYLLF5BOG6!L6-AFX1UD.JPG')
    img = np.asarray(bytearray(r.content), dtype='uint8')
    img = cv2.imdecode(img, cv2.IMREAD_COLOR)
    return img

def run_hog(img, model):
    df = location_to_df(
                    slideExtract(img, model, step = 112)
    )
    img = extract_im(df, img)
    return img
    

