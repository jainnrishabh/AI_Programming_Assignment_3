import numpy as np

def activation_relu(x):
    return np.maximum(0,x)

def derivative_relu(x):
    return 1. * ( x > 0 )

def derivative_sigmoid(x):
    return activation_sigmoid(x) * ( 1 - activation_sigmoid(x) )

def activation_sigmoid(x):
    return 1 / (1 + np.exp(-x))

def mse(y, y_predicted):
    return np.mean(np.square(y - y_predicted))

def derivative_mse_loss(target, predictions):
    return 2 * (predictions-target)/predictions.size

def classify(x, y, w1,w2,b1,b2):
    y1 = np.dot(x,w1) + b1
    a1 = activation_relu(y1)
    y2 = np.dot(a1,w2) + b2
    a2 = activation_relu(y2)

    loss =  mse(y, a2)

    dA2 = derivative_mse_loss(y,a2)
    dY2 = dA2 * derivative_relu(y2)
    dW2 = np.dot(a1.T, dY2)
    dB2 = np.sum(dY2 , axis = 0 , keepdims= True) 

    dA1 = np.dot(dY2,w2.T)
    dY1 = dA1 *  derivative_relu(y1)
    dW1 = np.dot(x.T,dY1)
    dB1 = np.sum(dY1 , axis = 0 , keepdims= True) 
    
    return dW1 , dW2 , dB1, dB2
    
def update_weights(w1, w2, b1, b2, dW1 , dW2 , dB1, dB2):
    learning_rate = 0.1
    w2 = w2 - learning_rate*dW2
    w1 = w1 - learning_rate*dW1
    b2 = b2 - learning_rate*dB2
    b1 = b1 - learning_rate*dB1
    return w1, w2, b1, b2

def predict(x,w1,w2,b1,b2):
    y1 = np.dot(x,w1) + b1
    a1 = activation_relu(y1)
    y2 = np.dot(a1,w2) + b2
    a2 = activation_relu(y2)
    return a2


def main():
    x = [[0,0],[0,1],[1,0],[1,1]]
    y = [[0,0],[0,1],[0,1],[1,0]]
    
    w1 = np.random.rand(2,2)
    b1 = np.random.rand(1,2)

    w2 = np.random.rand(2,2)
    b2 = np.random.rand(1,2)

    for i in range(0,1000):
        dW1 , dW2 , dB1, dB2 = classify(np.array(x),np.array(y),w1,w2,b1,b2)
        w1,w2,b1,b2 = update_weights(w1, w2, b1, b2, dW1, dW2, dB1, dB2)

    print(predict(np.array(x),w1,w2,b1,b2))

main()