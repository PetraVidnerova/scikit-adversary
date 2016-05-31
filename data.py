from keras.datasets import mnist 
import matplotlib.pyplot as plot 


def load_zeros():
    (X_train, Y_train), (X_test, Y_test) = mnist.load_data() 
    X_train = X_train.reshape(60000, 784) 
    X_test = X_test.reshape(10000, 784)

    ZERO_train = X_train[Y_train == 0] 
    ZERO_test = X_test[Y_test == 0] 

    return ZERO_train, ZERO_test 

def plot(x):
    X = x.reshape(28, 28) 
    


if __name__ == "__main__":
    ZERO_train, ZERO_test = load_zeros() 
    print(ZERO_train.shape) 
    print(ZERO_test.shape) 


