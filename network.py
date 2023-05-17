import numpy as np

# Part 3, 4, 5 
def activation_relu(x):                                             # function RELU(x)
    return np.maximum(0,x)                                          # return max(0,x)

def derivative_relu(x):                                             # function RELU'(x)
    return 1. * ( x > 0 )                                           # return 1 if x > 0 else 0

def derivative_sigmoid(x):                                          # derivative_relu
    return activation_sigmoid(x) * ( 1 - activation_sigmoid(x) )    # return sigmoid(x) * ( 1 - sigmoid(x) )

def activation_sigmoid(x):                                          # function SIGMOID(x)
    return 1 / (1 + np.exp(-x))                                     # return 1 / (1 + e^-x)

def activation_tanh(x):                                             # function TANH(x)
 return (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))               # return (e^x - e^-x) / (e^x + e^-x)

def derivative_tanh(x):                                             # function TANH'(x)              
    dt=1-activation_tanh(x)**2                                      # derivative_tanh                    
    return dt                                                       # return dt        

def mse(y, y_predicted):                                            # function MSE(y, y_predicted)
    return np.mean(np.square(y - y_predicted))                      # return mean((y - y_predicted)^2)

def derivative_mse_loss(target, predictions):                       # function MSE'(y, y_predicted)
    return 2 * (predictions-target)/predictions.size                # return 2 * (y - y_predicted) / y.size

def classify(x, y, w1,w2,b1,b2):                                    # function CLASSIFY(x, y, w1,w2,b1,b2)
    y1 = np.dot(x,w1) + b1                                          # y1 ← x.w1 + b1
    a1 = activation_tanh(y1)                                        # a1 ← SIGMOID(y1)
    y2 = np.dot(a1,w2) + b2                                         # y2 ← a1.w2 + b2
    a2 = activation_tanh(y2)                                        # a2 ← SIGMOID(y2)

    loss =  mse(y, a2)                                              # loss ← MSE(y, a2)

    dA2 = derivative_mse_loss(y,a2)                                 # dA2 ← MSE'(y, a2)
    dY2 = dA2 * derivative_tanh(y2)                                 # dY2 ← dA2 * SIGMOID'(y2)
    dW2 = np.dot(a1.T, dY2)                                         # dW2 ← a1T.dY2
    dB2 = np.sum(dY2 , axis = 0 , keepdims= True)                   # dB2 ← sum(dY2)

    dA1 = np.dot(dY2,w2.T)                                          # dA1 ← dY2.w2T
    dY1 = dA1 *  derivative_tanh(y1)                                # dY1 ← dA1 * SIGMOID'(y1)
    dW1 = np.dot(x.T,dY1)                                           # dW1 ← xT.dY1
    dB1 = np.sum(dY1 , axis = 0 , keepdims= True)                   # dB1 ← sum(dY1)

    return dW1 , dW2 , dB1, dB2                                     # return dW1 , dW2 , dB1, dB2
    
def update_weights(w1, w2, b1, b2, dW1 , dW2 , dB1, dB2):           # function UPDATE_WEIGHTS(w1, w2, b1, b2, dW1 , dW2 , dB1, dB2)
    learning_rate = 0.1                                             # learning_rate ← 0.1        
    w2 = w2 - learning_rate * dW2                                   # w2 ← w2 - learning_rate * dW2
    w1 = w1 - learning_rate * dW1                                   # w1 ← w1 - learning_rate * dW1    
    b2 = b2 - learning_rate * dB2                                   # b2 ← b2 - learning_rate * dB2
    b1 = b1 - learning_rate * dB1                                   # b1 ← b1 - learning_rate * dB1
    return w1, w2, b1, b2                                           # return w1, w2, b1, b2

def predict(x,w1,w2,b1,b2):                                         # function PREDICT(x,w1,w2,b1,b2)
    y1 = np.dot(x,w1) + b1                                          # y1 ← x.w1 + b1
    a1 = activation_tanh(y1)                                        # a1 ← SIGMOID(y1)
    y2 = np.dot(a1,w2) + b2                                         # y2 ← a1.w2 + b2
    a2 = activation_tanh(y2)                                        # a2 ← SIGMOID(y2)
    return a2                                                       # return a2


# for testing of neural network
# def main():
#     x = [[0,0],[0,1],[1,0],[1,1]]
#     y = [[0,0],[0,1],[0,1],[1,0]]
    
#     w1 = np.random.rand(2,2)
#     b1 = np.random.rand(1,2)

#     w2 = np.random.rand(2,2)
#     b2 = np.random.rand(1,2)

#     for i in range(0,1000):
#         dW1 , dW2 , dB1, dB2 = classify(np.array(x),np.array(y),w1,w2,b1,b2)
#         w1,w2,b1,b2 = update_weights(w1, w2, b1, b2, dW1, dW2, dB1, dB2)

#     print(predict(np.array(x),w1,w2,b1,b2))