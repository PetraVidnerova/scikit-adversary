import glob 
import numpy as np 
import pandas as pd 
from keras.models import model_from_json

NUMBER_NAME = "five"
NUMBER = 5

def npy2csv():
    """ Reads all files from matrices/*.npy and creates 
    one csv file with examples.
    
    """ 
    X = np.empty((0, 784))

    for file_name in glob.glob("matrices/*{}*.npy".format(NUMBER_NAME)):
        M = np.load(file_name)
        X = np.vstack((X, M)) 
        
    Y = np.array([0] * len(X))
    Y = Y.reshape(len(Y), 1)
    
    # Exclude those classified as NUMBER. 
    model = model_from_json(open("mlp.json").read()) 
    model.load_weights("mlp_weights.h5")
    yy = model.predict(X)
    labels = []   
    for vector in yy:
        labels.append(vector.argmax())
    labels = np.array(labels)
    print(labels)

    print(X.shape) 
    X = X[labels != NUMBER] 
    print(X.shape) 

#    Y = np.array([0] * len(X))

    np.save("data/adversary_inputs_{}".format(NUMBER_NAME), X) 
#    np.save("data/adversary_labels", Y) 


def load_adversary():
    """ Loads data from advesary_inputs.npy and adversary_labels.npy. 
        Returns data as X, y.

    """

    X = np.load("data/adversary_inputs_{}.npy".format(NUMBER_NAME)) 
#    y = np.load("data/adversary_labels.npy") 

    return X


def test(): 
    """ Load data from csv file. """ 
    
    X = load_adversary() 

    print(X.shape)
   



if __name__ == "__main__":

    npy2csv()
    test()
