import sys, os
import numpy as np
sys.path.append(os.pardir)
import deep_learning_scratch.dataset.mnist as m
from PIL import Image
import pickle
from workspace2 import numerical_gradient

def get_data():
    (x_train, t_train),(x_test, t_test) = m.load_mnist(normalize = True, flatten = True, one_hot_label = False)
    return x_train, t_train

def img_show(img):
    pil_img = Image.fromarray(img)
    pil_img.show()

def init_network():
    with open('/Users/yamashitashiori/Desktop/Python3/deep_learning_scratch/ch03/sample_weight.pkl','rb') as f:
        network = pickle.load(f)
    return network

def Softmax(x):
    return np.exp(x)/np.sum(np.exp(x))

def sigmoid(x):
    return 1/(1+np.exp(-x))


def predict(network,x):
    w1, w2, w3 = network['W1'], network['W2'], network['W3']
    b1,b2,b3 = network['b1'], network['b2'], network['b3']
    a1 = np.dot(x,w1)+b1
    z1 = sigmoid(a1)    
    a2 = np.dot(z1,w2)+b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2,w3)+b3
    y = Softmax(a3)
    return y

def cross_entropy_error(y,t):
    if y.ndim == 1:
        t = t.reshape(1,t.size)
        y = y.reshape(1,y.size)
    return -np.sum(t*np.log(y+(1e-7)))



x,t = get_data()
network = init_network()
batch_size = 100
accuracy_cnt = 0
for i in range(0,len(x),batch_size):
    x_batch = x[i:i+batch_size]
    y_batch = predict(network,x_batch)
    p = np.argmax(y_batch, axis = 1)
    accuracy_cnt += np.sum(p == t[i:i+batch_size])

print(accuracy_cnt/len(x))        
