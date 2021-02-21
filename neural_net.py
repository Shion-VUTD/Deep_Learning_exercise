#7層のNN生成
#batch_norm,Dropoutのレイヤ生成はとりあえずデフォルトのを使って後で自力で実装
import sys,os
sys.path.append('/Users/myname/Desktop/Python3')
from deep_learning_scratch_for_exercise.common.layers import Dropout,BatchNormalization,Sigmoid,Relu,Affine,SoftmaxWithLoss
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict


class MultiLayerNet_extend:
    #-----ハイパーパラメータ------#
        #imput_size
        #hidden_size_list
        #output_size
        #活性化関数の種類（Relu,Sigmoid)
        #初期値
        #--標準偏差
        #Dropoutの割合（そもそもするかどうか）
        #Batch_norm の初期値（そもそもするかどうか）
        
        #(忘れてました)
        #weight_decay_lambda


    #レイヤの保持
    #パラメータの保持
    #最初は、さっき挙げた要素を全部突っ込む
    def __init__(self,input_size,hidden_size_list,output_size,activation = 'relu',weight_init_std='relu',use_dropout = False,dropout_ratio = 0.5,use_batch_norm = False,weight_decay_lambda=0):
        #渡された値を脳死で保持していくっ（中間層の数も一緒に保持しとくとみやすい）
        self.input_size = input_size
        self.hidden_size_list = hidden_size_list
        self.output_size = output_size
        self.hidden_layer_num = len(hidden_size_list)
        self.activation = activation
        self.weight_init_std = weight_init_std
        self.use_dropout = use_dropout
        self.use_batch_norm = use_batch_norm
        self.weight_decay_lambda = weight_decay_lambda

        self.params = {}
        self.layers = OrderedDict()

        #重みの初期化
        self.__init_weight_init_std()

        #レイヤの実装
        activation_layer = {'Sigmoid':Sigmoid, 'Relu':Relu}
        for i in range(1,self.hidden_layer_num+1):
            #パラメータの挿入は後ほど初期化
            self.layers['Affine'+str(i)] = Affine(self.params['w'+str(i)],self.params['b'+str(i)])
            
            #Batch_normを使う場合、AffineレイヤとActivationレイヤの間に挿入することが多い
            if self.use_batch_norm:
                #パラメータの挿入
                self.params['gamma'+str(i)] = np.ones(hidden_size_list[i-1])
                self.params['beta'+str(i)] = np.zeros(hidden_size_list[i-1])
                #レイヤの挿入
                self.layers['BatchNorm'+str(i)] = BatchNormalization(self.params['gamma'+str(i)], self.params['beta'+str(i)])

            #Activationレイヤの挿入
            self.layers['Activation_Func'+str(i)] = activation_layer[self.activation]()

            #Dropoutレイヤの挿入
            if use_dropout:
                self.layers['Dropout'+str(i)] = Dropout(dropout_ratio)

        #最後の層は活性化関数がSoftmax
        self.layers['Affine'+str(self.hidden_layer_num+1)] = Affine(self.params['w'+str(self.hidden_layer_num+1)],self.params['b',str(self.hidden_layer_num+1)])
        
        self.lastLayer = SoftmaxWithLoss()

    #次に、初期値の設定
    
    def __init_weight_init_std(self,weight_init_std):
        all_size_list = [self.input_size]+self.hidden_size_list+[self.output_size]
        for i in range(1,self.hidden_layer_num+2): #全ての層について議論
            scale = weight_init_std
            if self.layers['Activation_Func'+str(i)] == Relu():
                scale = np.sqrt(2/self.hidden_size_list[i-1])
            else:
                scale = 1/np.sqrt(self.hidden_size_list[i-1])    

            self.params['w'+str(i)] = scale*np.random.randn(all_size_list[i-1],all_size_list[i])   
            self.params['b'+str(i)] = np.zeros(all_size_list[i])

    def predict(self,x,train_flg=False):
        for key,layer in self.layers.items():
            if 'Dropout' in key or 'BatchNorm' in key: #DropoutレイヤかBatchNormレイヤのいずれかである時
                x = layer.forward(x,train_flg)

            else:
                x = layer.forward(x)
        return x

    def loss(self,x,t,train_flg = False): 
        y = self.predict(x,train_flg)

        #weight_decayの実装
        weight_decay = 0
        for i in range(1,self.hidden_layer_num+2):
            weight_decay += 0.5*np.sum(self.params['w'+str(i)]**2)*self.weight_decay_lambda

        loss = self.lastLayer.forward(y,t) + weight_decay
        return loss    

    def accuracy(self,x,t,train_flg = False):
        #---後でーーー#
        y = self.predict(x,train_flg)
        p = np.argmax(y,axis = 1)
        if t.ndim != 1:
            t = np.argmax(t,axis = 1)
        return np.sum(p==t)/t.shape[0]

    def gradient(self,x,t,train_flg = False):
        self.loss(x,t,train_flg)
        dout = 1
        dout = self.lastLayer.backward(dout)
        layers_list = list(self.layers.values())
        layers_list.reverse()
        for layer in layers_list:
            dout = layer.backward(dout)

        #勾配の決定
        grads = {}    
        for i in range(1,self.hidden_layer_num+2):
            grads['w'+str(i)] = self.layers['Affine'+str(i)].dw + self.weight_decay_lambda*self.params['w'+str(i)]
            grads['b'+str(i)] = self.layers['Affine'+str(i)].dbeta

            #Batch_normレイヤのパラメータ（gamma,beta)を決定
            if self.use_batch_norm and i != self.hidden_layer_num+1: #BatchNormレイヤは出力層にはないので
                grads['gamma'+str(i)] = self.layers['BatchNorm'+str(i)].dgamma
                grads['beta'+str(i)] = self.layers['BatchNorm'+str(i)].dbeta

        return grads




        

                  

            



                








    


