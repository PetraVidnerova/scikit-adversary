import random 
import numpy as np
import matplotlib.pyplot as plot 

NUMBER_NAME = "five"

def show_random(X):

    x = random.choice(X) 
    x = x.reshape(28, 28) 
    plot.imshow(x, interpolation="none", cmap=plot.cm.Greys)
    plot.show()


def main():
    X = np.load("data/adversary_inputs_{}.npy".format(NUMBER_NAME)) 
    for i in range(10):
        show_random(X) 
 

if __name__ == "__main__": 
    main() 
    


