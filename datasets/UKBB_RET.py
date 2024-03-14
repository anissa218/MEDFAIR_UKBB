import torch
import numpy as np
from PIL import Image
import pickle
from datasets.BaseDataset import BaseDataset
import cv2

class UKBB_RET(BaseDataset):
    def __init__(self, dataframe, path_to_pickles, sens_name, sens_classes, transform):
        super(UKBB_RET, self).__init__(dataframe, path_to_pickles, sens_name, sens_classes, transform)

        """
            Dataset class for UKBB retinal images dataset.
            
            Arguments:
            dataframe: the metadata in pandas dataframe format.
            path_to_pickles: path to the pickle file containing images.
            sens_name: which sensitive attribute to use, e.g., Sex or Age_binary or Ethnicity(others not defined, and Age_multi has 3 classes instead of 4)
            sens_classes: number of sensitive classes. Depends on attribute
            transform: whether conduct data transform to the images or not.
            
            Returns:
            index, image, label, and sensitive attribute.
        """
        
        # # if you're using the pickled images
        # with open(path_to_pickles, 'rb') as f: 
        #     self.tol_images = pickle.load(f)
            
        self.A = self.set_A(sens_name)
        self.Y = (np.asarray(self.dataframe['binaryLabel'].values) > 0).astype('float')
        self.AY_proportion = None
    
    def __getitem__(self, idx):
        # get the item based on the index
        item = self.dataframe.iloc[idx]
            
        img = cv2.imread(item['image_path'],cv2.IMREAD_GRAYSCALE) #so it only has one channel
    
        img = cv2.resize(img, (256, 256))
        img = Image.fromarray(img).convert('RGB') # converts image from 2 channel to 3 channel

        # # if you're using pickled images:
        # img = Image.fromarray(self.tol_images[idx]).convert('RGB')

        img = self.transform(img) # maybe should be doing different transforms for this but will leave for now
        
        label = torch.FloatTensor([int(item['binaryLabel'])])
        
        # get sensitive attributes in numerical values
        sensitive = self.get_sensitive(self.sens_name, self.sens_classes, item)
                               
        return img, label, sensitive, idx