#とりあえずaccuracy以外何も見ないで実装してみた（練習）

import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('/Users/yamashitashiori/Desktop/Python3/Deep_Learning_exercise')
from collections import OrderedDict

def softmax(x):
    if x.ndim != 1:
        return np.exp(x)/(np.sum(np.exp(x),axis = 1)+1e-7)
    else:
        return np.exp(x)/(np.sum(np.exp(x))+1e-7)   
    
def cross_entropy_error(y,t): 
    if y.ndim == 1:
        y = y.reshape(1,-1)
        t = t.reshape(1,-1)
    return -np.sum((t*np.log(y+1e-7)))/t.shape[0]


class Affine:
    def __init__(self,w,b):
        self.x = None
        self.db = None
        self.dw = None
        self.w = w
        self.b = b

    def forward(self,x):
        self.x = x
        out = np.dot(x,self.w)+self.b
        return out

    def backward(self,dout):
        dx = np.dot(dout,self.w.T)
        self.dw = np.dot(self.x.T,dout)
        self.db = np.sum(dout,axis = 0)
        return dx

class BatchNorm:
    def __init__(self,gamma,beta,momentum= 0.9,running_mean= None,running_var=None):
        #値を保持しとかなきゃいけないやつはどれ？
        self.gamma = gamma
        self.beta = beta
        self.running_mean = running_mean
        self.running_var = running_var
        self.momentum = momentum

        #逆伝播
        self.dgamma = None
        self.dbeta = None
        self.xc = None
        self.std = None
        self.xn = None
        self.input_size = None

    def __forward(self,x,train_flg=True):  #xが2次元の場合
        self.input_size = x.shape[0]
        if train_flg:
            #初期値の設定
            self.running_mean = np.zeros(self.input_size)
            self.running_var = np.zeros(self.input_size)

            #順伝播
            mu = np.mean(x,axis = 0)
            xc = x-mu
            var = np.sum(xc,axis = 0)/self.input_size
            std = np.sqrt(var)
            xn = xc/std
        
            #逆伝播に使う値を格納
            self.xc = xc
            self.std = std
            self.xn = xn
            self.running_mean = self.momentum*self.running_mean + (1-self.momentum)*mu
            self.running_var = self.momentum*self.running_var + (1-self.momentum)*var
            
            return self.gamma*xn+self.beta

        else:
            mu_test = self.running_mean
            var_test = self.running_var
            xc_test = mu_test/np.sqrt(var_test)
            return self.gamma*xc_test + self.beta


    def forward(self,x,train_flg=True):   #xが2次元でない場合も含む
        if x.ndim != 2:
            shape_init = x.shape
            n = x.shape[0]
            x = x.reshape(n,-1)

        out = self.__forward(x,train_flg=True)    
        out = out.reshape(shape_init)
        return out

    def backward(self,dout):
        self.dbeta = np.sum(dout,axis = 0)
        dxn = dout*self.gamma
        self.dgamma = dout*self.xn
        dxc1 = dxn/self.std
        dstd = -dxn*self.xc/(self.std*self.std)
        dvar = 0.5*dstd/self.std
        dxc2 = 2*self.xc*dvar/self.input_size
        dx1 = dxc1+dxc2
        dmu = -dx1
        dx2 = dmu/self.input_size
        dx = dx1+dx2
        return dx

class Sigmoid:
    def __init__(self):
        self.out = None
    def forward(self,x):
        out = 1/(1+np.exp(-x))
        self.out = out
        return out
    def backward(self,dout):
        return dout*self.out*(1-self.out)

class Relu:
    def __init__(self):
        self.mask = None
    def forward(self,x):
        self.mask = (x<=0)
        x[self.mask] = 0
        return x
    def backward(self,dout):
        dout[self.mask] = 0
        return dout                    

class Dropout:
    def __init__(self,dropout_ratio=0.5):
        self.dropout_ratio = dropout_ratio
        self.mask = None
    def forward(self,x,train_flg=True):
        if train_flg:
            self.mask = np.random.rand(x.shape)>self.dropout_ratio
            x[self.mask] = 0
            return x

        else:
            x *= (1-self.dropout_ratio)    
            return x

    def backward(self,dout):
        dout[self.mask] = 0
        return dout


class SoftmaxWithLoss:
    def __init__(self):
        self.y = None
        self.t = None
    def forward(self,x,t):  #softmaxやってlossを計算
        y = softmax(x)
        self.y = y
        self.t = t
        loss = cross_entropy_error(y,t)
        return loss
    def backward(self,dout=1): #backwardに渡す引数をdoutだけにしたいからtもクラス内に格納しておく
        return dout*(self.y-self.t)/self.t.shape[0]   #batch_size分だけ分かれて逆伝播するからbatch_sizeでわる！


#n層のNNを定義
class MultiLayerNet:
    #-------必要なもの--------
    #パラメータ格納庫
    #初期値の標準偏差
    #入力層のサイズ
    #中間層の数、サイズ
    #出力層のサイズ
    #活性化関数
    #BatchNorm使うか
    #Dropout使うか
    #Drououtの割合
    #weightdecayの重み
    #------------------------

    def __init__(self,input_size,hidden_size_list,output_size,weight_init_std,activation='Relu',use_batch_norm=False,use_dropout=False,dropout_ratio=0.5,weight_decay_lambda=0):
        self.input_size = input_size
        self.hidden_size_list = hidden_size_list
        self.hidden_layer_num = len(hidden_size_list)
        self.output_size = output_size
        self.activation = activation
        self.weight_init_std = weight_init_std
        self.use_batch_norm = use_batch_norm
        self.use_dropout = use_dropout
        self.weight_decay_lambda = weight_decay_lambda
        self.dropout_ratio = dropout_ratio

        self.params = {}
        self.layers = OrderedDict()
        activation_layer = {'Relu':Relu,'Sigmoid':Sigmoid}

        #初期値は別の関数に分けて決めますよってこと
        self.__init_weight()

        #層の格納
        for i in range(1,self.hidden_layer_num+2):
            self.layers['Affine'+str(i)] = Affine(self.params['w'+str(i)],self.params['b'+str(i)])
            if use_batch_norm and i != self.hidden_layer_num+1:  #batch_noemがOKで最終層じゃない
                self.params['gamma'+str(i)] = np.ones(hidden_size_list[i-1])
                self.params['beta'+str(i)] = np.zeros(hidden_size_list[i-1])
                self.layers['BatchNorm'+str(i)] = BatchNorm(self.params['gamma'+str(i)],self.params['beta'+str(i)])
            
            self.layers['Activation_func'] = activation_layer[self.activation]()

            if use_dropout and i != self.hidden_layer_num+1:
                self.layers['Dropout'] = Dropout(self.dropout_ratio)

            self.lastLayer = SoftmaxWithLoss()    
            
    #パラメータの初期化        
    def __init_weight(self,weight_init_std):
        all_size_list = [self.input_size]+self.hidden_size_list+[self.output_size]
        scale = weight_init_std
        for i in range(1,self.hidden_layer_num+2):
            if self.activation == 'Relu':
                scale = np.sqrt(2/all_size_list[i-1])
            elif self.activation == 'Sigmoid':
                scale = 1/np.sqrt(all_size_list[i-1])   
            self.params['w'+str(i)] = scale*np.random.randn(all_size_list[i-1],all_size_list[i])     
            self.params['b'+str(i)] = np.zeros(all_size_list[i])   


    #推測
    def predict(self,x,train_flg=True):
        for key,layer in self.layers.items():
            if 'BatchNorm' in key or 'Dropout' in key:
                x = layer.forward(x,train_flg)
            else:
                x = layer.forward(x)    

        return x

    def loss(self,x,t,train_flg=True):
        y = self.predict(x,train_flg)
        weight_decay = 0
        for i in range(1,self.hidden_layer_num+2):
            weight_decay += 0.5*self.weight_decay_lambda*np.sum(self.params['w'+str(i)])
        
        return self.lastLayer.forward(y,t)+weight_decay

    def gradient(self,x,t):
        loss = self.loss(x,t,train_flg=True)    
        #逆伝播
        dout = 1
        dout = dout*SoftmaxWithLoss.backward(dout)
        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        #勾配決定
        grads = {}
        for i in range(1,self.hidden_layer_num+2):
            grads['w'+str(i)] = self.layers['Affine'+str(i)].dw 
            grads['b'+str(i)] = self.layers['Affine'+str(i)].db
            if self.use_batch_norm and i != self.hidden_layer_num+1:
                grads['gamma'+str(i)] = self.layers['BatchNorm'+str(i)].dgamma
                grads['beta'+str(i)] = self.layers['BatchNorm'+str(i)].dbeta

        return grads

    def accuracy(self,x,t):
        y = self.predict(x,train_flg = True)
        if y.ndim == 1:
            y = y.reshape(1,-1)
            t = t.reshape(1,-1)    
        p = np.argmax(y,axis = 1)
        t = np.argmax(t,axis = 1)
        return np.sum(p==t)/y.shape[0]






            

        
                



        








