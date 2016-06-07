import numpy as np 
from keras.utils import np_utils 
from keras.models import model_from_json 

NUMBER = 5
NUMBER_NAME = "five" 

def class_accuracy(target, predicted):
    correct = 0
    for t, p in zip(target, predicted):
        print(t.argmax(), p.argmax())
        if t.argmax() == p.argmax():
            correct += 1 
    return correct / len(target)

def main():
    # Load data. 
    X = np.load("data/adversary_inputs_{}.npy".format(NUMBER_NAME)) 
    y = np.array([ NUMBER] * len(X))
    
    print(y)

    Y = np_utils.to_categorical(y, 10)

    # Load mlp. 
    model = model_from_json(open("mlp.json").read()) 
    model.load_weights("mlp_weights.h5")

    # Eval data. 
    yy = model.predict(X) 

    # Compute accuracy.
    acc = class_accuracy(yy, Y) 

    print("Class. accuracy {}".format(acc))


if __name__ == "__main__":
    main() 
