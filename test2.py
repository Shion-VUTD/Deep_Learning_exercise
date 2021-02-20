import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
import sys,os
sys.path.append('/Users/yamashitashiori/Desktop/Python3')
from deep_learning_scratch_for_exercise.dataset_zero import mnist 

#関数の定義
def softmax(x):
    m = np.max(x,axis = -1,keepdims = True)
    return np.exp(x-m)/np.sum(np.exp(x-m)+1e-7,axis = 1,keepdims = True)

def cross_entropy_error(y,t):
    return -np.sum(t*np.log(y))/t.shape[0]

def sigmoid(x):
    return 1/(1+np.exp(-x))

def numerical_gradient(f,x,h= 1e-4):
    grad = np.zeros_like(x)
    h = 1e-4
    i = x.shape[0]
    if x.ndim == 2:
        j = x.shape[1]
    else:
        j = 0    
    x.flatten()
    for k in range(len(x)):
        idx = x[i]
        x[i]= idx + h
        f1x = f(x)
        x[i] = idx-h
        f2x = f(x)
        grad[i] = (f1x - f2x) / (2*h)
        x[i] = idx

    if j != 0:
        grad.reshape(i,j)
    
    return grad   


#レイヤの定義
class Affine:
    def __init__(self,w,b):
        self.w = w
        self.b = b
        self.x = None
        self.dw = None
        self.db = None

    def forward(self,x):
        self.x = x
        out = np.dot(x,self.w)+self.b
        return out

    def backward(self,dout):
        self.dw = np.dot(self.x.T, dout)
        dx = np.dot(dout,self.w.T)
        self.db = np.sum(dout,axis = 0)
        return dx

class Sigmoid:
    def __init__(self):
        self.out = None

    def forward(self,x):
        out = sigmoid(x)
        self.out = out
        return out

    def backward(self,dout):
        dx = dout*(1-self.out)*self.out
        return dx   

class Relu:
    def __init__(self):
        self.mask = None
    def forward(self,x):
        self.mask = (x <= 0)   
        out = x.copy() #xの変更に伴ってself.maskが変更されてしまうのを防ぐ
        out[self.mask] = 0        
        return out      
    def backward(self,dout):
        dout[self.mask] = 0
        dx = dout
        return dx    


class SoftmaxWithLoss:
    def __init__(self):
        self.y = None
        self.t = None

    def forward(self,x,t):
        self.t = t
        y = softmax(x)
        self.y = y
        out = cross_entropy_error(y,t)/self.t.shape[0]
        return out

    def backward(self,dout):
        dx = self.y - self.t
        return dx

#最適化手法の定義
class SGD:
    def __init__(self,lr = 0.01):
        self.lr = lr

    #パラメータの更新    
    def upgrade(self,params,grads):
        for key in params.keys():
            params[key] -= self.lr*grads[key]


        

#誤差逆伝播のニューラルネット
class TwoLayorNet:
    def __init__(self,input_size,hidden_size,output_size,weight_init_std = 0.01):
        #パラメータを保持
        self.params = {}
        self.params['w1'] = np.random.randn(input_size,hidden_size)*weight_init_std
        self.params['b1'] = np.zeros(hidden_size)
        self.params['w2'] = np.random.randn(hidden_size,output_size)*weight_init_std
        self.params['b2'] = np.zeros(output_size)

        #レイヤを保持
        self.layers = OrderedDict()
        self.layers['Affine1'] = Affine(self.params['w1'],self.params['b1'])
        self.layers['Relu1'] = Relu()
        self.layers['Affine2'] = Affine(self.params['w2'],self.params['b2'])
        
        self.lastLayer = SoftmaxWithLoss()

    def predict(self,x):
        for layer in self.layers.values():
            x = layer.forward(x)
        return x

    def loss(self,x,t):
        y = self.predict(x)
        lost = self.lastLayer.forward(y,t)        
        return lost

    def loss_with_weight_decay(self,x,t):
        loss = self.loss(x,t)
        weight = 0
        for param in self.params.keys():
            weight += (0.1*np.sum(self.params[param]**2))/2
        return loss + weight    

    def numerical_gradients(self,x,t):
        loss_w = lambda w: self.loss(x,t)
        grads = {}
        for key in self.params.keys():
            grads[key] = numerical_gradient(loss_w,self.params[key])

        return grads

    def gradient(self,x,t,weight_decay=False):
        layers = list(self.layers.values())
        layers.reverse()
        dout = 1
        dout = self.lastLayer.backward(dout)
        for layer in layers:
            dout = layer.backward(dout)

        grads = {}
        if weight_decay:
            grads['w1'] = self.layers['Affine1'].dw+self.params['w1']*0.1
            grads['b1'] = self.layers['Affine1'].db
            grads['w2'] = self.layers['Affine2'].dw+self.params['w2']*0.1
            grads['b2'] = self.layers['Affine2'].db
        else:
            grads['w1'] = self.layers['Affine1'].dw
            grads['b1'] = self.layers['Affine1'].db
            grads['w2'] = self.layers['Affine2'].dw
            grads['b2'] = self.layers['Affine2'].db


        return grads

    def accuracy(self,x,t):
        y = self.predict(x)
        p = np.argmax(y,axis = 1)
        if t.ndim != 1:
            t = np.argmax(t,axis = 1)
        return np.sum(p==t)/t.shape[0]  



#データセットの準備
(x_train,t_train),(x_test,t_test) = mnist.load_mnist(normalize = True,one_hot_label=True)
train_size = x_train.shape[0]
batch_size = 100
epochs = 10
iters_per_epoch = train_size//batch_size
optimizer = SGD(lr=0.01)
network = TwoLayorNet(input_size = 784,hidden_size = 50, output_size=10)
train_loss_list = []
train_acc_list = []
test_acc_list = []

#学習
for epoch in range(epochs):
    for iter in range(iters_per_epoch):
        #バッチの生成
        batch_mask = np.random.choice(train_size,batch_size)
        x_batch = x_train[batch_mask]
        t_batch = t_train[batch_mask]

        #勾配の取得
        lost = network.loss(x_batch,t_batch)
        grads = network.gradient(x_batch,t_batch)

        #最適化手法の実行
        optimizer.upgrade(network.params,grads)


    #精度検証
    acc_train = network.accuracy(x_train,t_train)
    acc_test = network.accuracy(x_test,t_test)
    print('epoch:',epoch,'accuracy_train:',acc_train,'accuracy_test:',acc_test)
    train_acc_list.append(acc_train)
    test_acc_list.append(acc_test)
    train_loss_list.append(lost)


#描画
plt.plot(train_loss_list)
plt.xlabel('epochs')
plt.ylabel('loss')
plt.show()

plt.plot(train_acc_list,'b')
plt.plot(test_acc_list,'r') 
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend()   
plt.show()
    

        








        


