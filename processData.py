from scipy.io import arff
import os 
import sys
import pandas as pd 
import numpy as np
import sklearn

DIRECTORY_PATH = os.path.dirname(os.path.realpath(__file__))

class DataObj(object):
    def __init__(self, file_name):
        self.file_path = os.path.join(DIRECTORY_PATH, file_name)
        self.full_data_set = None
        self.meta_information = None
        self.x_train = None
        self.y_train = None
        self.x_test= None
        self.y_test = None

class PhishingData(DataObj):
    def load(self, split_percentage=0.2):
        try:
            data, self.meta_information = arff.loadarff(self.file_path)
            self.full_data_set = pd.DataFrame(data)
        except:
            print("Error loading data set, check path information")
            sys.exit(0)
            
        y_data = self.full_data_set["Result"]
        y_data=y_data.astype('int')

        x_data = self.full_data_set.loc[:, self.full_data_set.columns != 'Result']
        x_data=x_data.astype('int')
        
        from sklearn.model_selection import train_test_split
        self.x_train, self.x_test, self.y_train, self.y_test =  train_test_split(x_data, y_data, test_size=split_percentage)
        

    


