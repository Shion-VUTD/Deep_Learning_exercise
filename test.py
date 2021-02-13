import numpy as np
import matplotlib.pylab as plt
from deep_learning_scratch.dataset import mnist

def sigmoid(x):
    return 1/(1+np.exp(-x))


def cross_entropy_error(y,t):
    if y.ndim == 1:
        t = t.reshape(1,t.size)
        y = y.reshape(1,y.size)
    return -np.sum(t*np.log(y+(1e-7)))

def Softmax(x):
    return np.exp(x)/np.sum(np.exp(x))
    
def numerical_gradient(f,init_x):

    xdash = init_x
   
  
    h = 1e-4
    i = xdash.shape[0]

    if xdash.ndim == 2:
        j = xdash.shape[1]
    else:
        j = 0    
    xdash = xdash.flatten()
   
    grad = np.zeros_like(xdash)
    for k in range(len(xdash)):
       
        idx = xdash[k]
        xdash[k]= idx + h
        f1x = f(xdash)
        xdash[k] = idx-h
        f2x = f(xdash)
        grad[k] = (f1x - f2x) / (2*h)
        xdash[k] = idx
          
    
    if j != 0:
        grad = grad.reshape(i,j)
        
      
    return grad

class TwoLayerNet:
    def __init__(self,input_size,hidden_size,output_size,weight_init_std=0.01):
        self.params = {}
        self.params['w1'] = np.random.randn(input_size,hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['w2'] = np.random.randn(hidden_size,output_size)
        self.params['b2'] = np.zeros(output_size)

    def predict(self,x):
        
        w1,w2 = self.params['w1'],self.params['w2']
        b1,b2 = self.params['b1'],self.params['b2']
        a1 = np.dot(x,w1)+b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1,w2)+b2
        y = Softmax(a2)
        return y

    def loss(self,x,t):
        y = self.predict(x)
        return cross_entropy_error(y,t)
        


    def numerical_gradients(self,x,t):
        
        w1,w2 = self.params['w1'],self.params['w2']
        b1,b2 = self.params['b1'],self.params['b2']
        loss_w = lambda w: self.loss(x,t)
        grads = {}
        grads['w1'] = numerical_gradient(loss_w,w1)
        grads['W2'] = numerical_gradient(loss_w,w2)
        grads['b1'] = numerical_gradient(loss_w,b1)
        grads['b2'] = numerical_gradient(loss_w,b2)

        return grads

    def accuracy(self,x,t):
        y = self.predict(x)
        p = np.argmax(y,axis=1)
        t = np.argmax(t,axis=1)
        return np.sum(p==t,dtype=np.int) / x.shape[0]

#学習
network = TwoLayerNet(input_size = 784,hidden_size=50,output_size= 10) 
(x_train,t_train),(x_test,t_test) = mnist.load_mnist(normalize=True,one_hot_label=True)
train_size = x_train.shape[0]
batch_size = 100
step_num = 100
learning_rate = 0.1
steps = []
losses = []
acc = []

for i in range(step_num):
    steps.append(i)
    batch_mask = np.random.choice(train_size,batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]
    
    #更新ごとに予測してグラフを描画
    losses.append(network.loss(x_batch,t_batch)/batch_size)
    acc.append(network.accuracy(x_batch,t_batch))


    #勾配ベクトルを求める
    grads = network.numerical_gradients(x_batch,t_batch)
    
    #パラメータの更新
    for param in ('w1','w2','b1','b2'):
        network.params[param] -= learning_rate * network.params[param]

plt.plot(steps,losses)
plt.plot(steps,acc)        
plt.legend()
plt.show()



    
        






