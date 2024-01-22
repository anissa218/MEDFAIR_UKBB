import pandas as pd
import numpy as np
import os
from PIL import Image
import h5py
import cv2
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.model_selection import train_test_split
import pickle
import time

root = '/gpfs3/well/papiez/users/hri611/python/MEDFAIR-PROJECT/MEDFAIR/'


val_meta = pd.read_csv(os.path.join(root,'data/mimic-cxr/splits/val.csv'))
path = os.path.join(root, 'data/mimic-cxr/pkls/')

images = []
for i in range(len(val_meta)):
    
    img = cv2.imread(val_meta.iloc[i]['path'],cv2.IMREAD_GRAYSCALE) #so it only has one channel
    # resize to the input size in advance to save time during training
    img = cv2.resize(img, (256, 256))
    images.append(img)

with open(path + 'val_images.pkl', 'wb') as f:
    pickle.dump(images, f)