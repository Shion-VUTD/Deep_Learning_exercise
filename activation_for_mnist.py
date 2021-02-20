from test2 import Affine,Relu,Sigmoid,SoftmaxWithLoss,SGD
from activation import activate
import numpy as np
import matplotlib.pyplot as plt
from deep_learning_scratch_for_exercise.dataset import mnist
from collections import OrderedDict

class FiveLayorNet:
    def __init__(self,input_size,hidden_size,output_size,weight_init_std_input = 0.01,weight_init_std_hidden = 0.01):
        #パラメータを保持
        self.params = {}
        self.params['w1'] = np.random.randn(input_size,hidden_size)*weight_init_std_input
        self.params['b1'] = np.zeros(hidden_size)
        self.params['w2'] = np.random.randn(hidden_size,hidden_size)*weight_init_std_hidden
        self.params['b2'] = np.zeros(hidden_size)
        self.params['w3'] = np.random.randn(hidden_size,hidden_size)*weight_init_std_hidden
        self.params['b3'] = np.zeros(hidden_size)
        self.params['w4'] = np.random.randn(hidden_size,hidden_size)*weight_init_std_hidden
        self.params['b4'] = np.zeros(hidden_size)
        self.params['w5'] = np.random.randn(hidden_size,output_size)*weight_init_std_hidden
        self.params['b5'] = np.zeros(output_size)


        #レイヤを保持
        self.layers = OrderedDict()
        self.layers['Affine1'] = Affine(self.params['w1'],self.params['b1'])
        self.layers['Relu1'] = Relu()
        self.layers['Affine2'] = Affine(self.params['w2'],self.params['b2'])
        self.layers['Relu2'] = Relu()
        self.layers['Affine3'] = Affine(self.params['w3'],self.params['b3'])
        self.layers['Relu3'] = Relu()
        self.layers['Affine4'] = Affine(self.params['w4'],self.params['b4'])
        self.layers['Relu4'] = Relu()
        self.layers['Affine5'] = Affine(self.params['w5'],self.params['b5'])
    
        
        self.lastLayer = SoftmaxWithLoss()

    def predict(self,x):
        for layer in self.layers.values():
            x = layer.forward(x)
        return x

    def loss(self,x,t):
        y = self.predict(x)
        lost = self.lastLayer.forward(y,t)        
        return lost

    def gradient(self,x,t):
        layers = list(self.layers.values())
        layers.reverse()
        dout = 1
        dout = self.lastLayer.backward(dout)
        for layer in layers:
            dout = layer.backward(dout)

        grads = {}
        grads['w1'] = self.layers['Affine1'].dw
        grads['b1'] = self.layers['Affine1'].db
        grads['w2'] = self.layers['Affine2'].dw
        grads['b2'] = self.layers['Affine2'].db
        grads['w3'] = self.layers['Affine3'].dw
        grads['b3'] = self.layers['Affine3'].db
        grads['w4'] = self.layers['Affine4'].dw
        grads['b4'] = self.layers['Affine4'].db
        grads['w5'] = self.layers['Affine5'].dw
        grads['b5'] = self.layers['Affine5'].db

        return grads

    def accuracy(self,x,t):
        y = self.predict(x)
        p = np.argmax(y,axis = 1)
        if t.ndim != 1:
            t = np.argmax(t,axis = 1)
        return np.sum(p==t)/t.shape[0]  




network_default = FiveLayorNet(input_size = 784,hidden_size = 50,output_size = 10)
network_Xavier = FiveLayorNet(input_size = 784,hidden_size = 50,output_size = 10, weight_init_std_input=np.sqrt(1/784),weight_init_std_hidden=np.sqrt(1/50))
network_He = FiveLayorNet(input_size = 784,hidden_size = 50,output_size = 10, weight_init_std_input=np.sqrt(2/784),weight_init_std_hidden=np.sqrt(2/50))
optimizer = SGD()
(x_train,t_train),(x_test,t_test) = mnist.load_mnist(normalize=True,one_hot_label=True)
epochs = 4
train_size = x_train.shape[0]
batch_size = 100
iters_for_epoch = train_size//batch_size
loss_for_default = []
loss_for_Xavier = []
loss_for_He = []

#学習
for epoch in range(epochs):
    for iter in range(iters_for_epoch):
        batch_mask = np.random.choice(train_size,batch_size)
        x_batch = x_train[batch_mask]
        t_batch = t_train[batch_mask]
        
        loss_default = network_default.loss(x_batch,t_batch)
        loss_Xavier = network_Xavier.loss(x_batch,t_batch)
        loss_He = network_He.loss(x_batch,t_batch)
        
        grads_default = network_default.gradient(x_batch,t_batch)
        grads_Xavier = network_Xavier.gradient(x_batch,t_batch)
        grads_He = network_He.gradient(x_batch,t_batch)

        optimizer.upgrade(network_default.params,grads_default)
        optimizer.upgrade(network_Xavier.params,grads_Xavier)
        optimizer.upgrade(network_He.params,grads_He)

    
        loss_for_default.append(loss_default)
        loss_for_Xavier.append(loss_Xavier)
        loss_for_He.append(loss_He)
    print('epoch:',epoch,'loss_default:',loss_default,'loss_Xavier:',loss_Xavier,'loss_He:',loss_He)

#描画
plt.plot(loss_for_default,label = 'default')
plt.plot(loss_for_Xavier,label = 'Xavier')
plt.plot(loss_for_He,label = 'He')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend()
plt.show()






