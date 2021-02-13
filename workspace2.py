import numpy as np
import matplotlib.pylab as plt


def func1(x):
    return 0.01*x[0]**2 +0.1*x[1]

def numerical_diff(f,x):
    h = 1e-4
    return (f(x+h)-f(x-h))/(2*h)

def sigmoid(x):
    return 1/(1+np.exp(-x))    


def numerical_gradient(f,x):
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

def gradient_descent(f,init_x,lr,step_num):
    x = init_x
    x_history = []
    for i in range(step_num):
        x_history.append(x.copy())
        grad = numerical_gradient(f,x)   
        x -= lr*grad
        
    return x,np.array(x_history)

def Softmax(x):
    return np.exp(x)/np.sum(np.exp(x))    

def func2(x):
    return x[0]**2+x[1]**2

def cross_entropy_error(y,t):
    if y.ndim == 1:
        t = t.reshape(1,t.size)
        y = y.reshape(1,y.size)
    return -np.sum(t*np.log(y+(1e-7)))


init_x = np.array([-3.0,4.0])
lr = 0.1
step_num = 100

x,x_history = gradient_descent(func2,init_x,lr,step_num)
print(x)
plt.plot( [-5, 5], [0,0], '--b')
plt.plot( [0,0], [-5, 5], '--b')
plt.plot(x_history[:,0], x_history[:,1], 'o')

plt.xlim(-3.5, 3.5)
plt.ylim(-4.5, 4.5)
plt.xlabel("X0")
plt.ylabel("X1")


x = np.array([0.6,0.9])
t = np.array([0,0,1])

class simpleNet:
    def __init__(self):
        self.w = np.random.randn(2,3)
    def prediction(self,x):
        return np.dot(x,self.w)
    def loss(self,x,t):
        z = self.prediction(x)    
        y = Softmax(z)
        return cross_entropy_error(y,t)

net = simpleNet()        

def f(w):
    return net.loss(x,t)    

dw = numerical_gradient(f,net.w)
print(dw)


