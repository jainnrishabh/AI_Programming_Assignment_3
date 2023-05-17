import numpy as np
from minmax import build_policy_table
from network import *
from hexapawn import *

def train_neural_network(x, y):
    w1 = np.random.uniform(-1, 1, size = (10,16))
    
    b1 = np.random.uniform(-1, 1, size = (1,16))

    w2 = np.random.uniform(-1, 1, size = (16,9))
    b2 = np.random.uniform(-1, 1, size = (1,9))

    for i in range(1000):
        dW1 , dW2 , dB1, dB2 = classify(np.array(x),np.array(y),w1,w2,b1,b2)
        w1,w2,b1,b2 = update_weights(w1, w2, b1, b2, dW1, dW2, dB1, dB2)

    for _ in range(10):
        idx = np.random.randint(0,len(x))
        y_pred = predict(np.array(x[idx]),w1,w2,b1,b2)
        print("Input: ",x[idx])
        print("Output: ",y[idx])
        print("Predicted: ",y_pred[0])   
        arr = []
        for data in y_pred[0]:
            i = 0
            if data > 0.6 and data < 1 : arr.append(1)
            elif data > -0.36 and data < 0.6 : arr.append(0)
            else : arr.append(-1)
            i += 1
        print("Normalizing Predicted: ",arr)   

hexBoard = HexaPawn(initial_board)
X, y = build_policy_table()
train_neural_network(X, y)
