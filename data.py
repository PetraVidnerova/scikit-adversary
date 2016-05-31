from keras.datasets import mnist 
import matplotlib.pyplot as plot 
import random 

def load_zeros():
    (X_train, Y_train), (X_test, Y_test) = mnist.load_data() 
    X_train = X_train.reshape(60000, 784) 
    X_test = X_test.reshape(10000, 784)

    ZERO_train = X_train[Y_train == 0] 
    ZERO_test = X_test[Y_test == 0] 

    return ZERO_train, ZERO_test 

def plot_example(x):
    X = x.reshape(28, 28) 
    plot.imshow(X, interpolation="none", cmap=plot.cm.Greys)
    plot.show() 


if __name__ == "__main__":
    ZERO_train, ZERO_test = load_zeros() 
    print(ZERO_train.shape) 
    print(ZERO_test.shape) 

    for x in range(10):
        plot_example(random.choice(ZERO_train)) 
    
    for x in range(10):
        plot_example(random.choice(ZERO_test))
    
